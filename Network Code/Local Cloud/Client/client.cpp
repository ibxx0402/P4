// Client side implementation of UDP client-server model 
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <vector>
#include <queue>
#include <map>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <fcntl.h>  // Add this at the top
#include <iomanip>   // For std::setprecision
#include <deque>     // For std::deque

// FFmpeg includes
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#define SERVER_IP "192.168.0.112"
#define PORT 9995
#define CLIENT_PORT 9998
#define MAXLINE 1400 // Max UDP packet size

struct FFmpegContext {
    const AVCodec* codec;
    AVCodecContext* context;
    AVFrame* frame_yuv;
    AVFrame* frame_bgr;
    SwsContext* sws_ctx;
    uint8_t* bgr_buffer;  // Persistent buffer for BGR conversion
    int bgr_buffer_size;
};

// Structure to hold frame reconstruction data.
struct FrameData {
    std::vector<uint8_t> data;
    size_t expected_size;
    uint32_t timestamp;
    bool complete;
};

// Add FPS tracking variables
double fps = 0.0;
int64_t prev_frame_time = 0;
const int FPS_SMOOTHING = 10; // Number of frames to average
std::deque<double> fps_history;

int main() {
    // Initialize FFmpeg
    FFmpegContext m_ffmpeg = {};
    avformat_network_init();
    m_ffmpeg.codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!m_ffmpeg.codec) {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }
    m_ffmpeg.context = avcodec_alloc_context3(m_ffmpeg.codec);
    if (!m_ffmpeg.context) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }
    
    // Add error resilience flags
    m_ffmpeg.context->err_recognition = AV_EF_CAREFUL;
    m_ffmpeg.context->flags |= AV_CODEC_FLAG_LOW_DELAY;
    m_ffmpeg.context->flags2 |= AV_CODEC_FLAG2_CHUNKS;
    
    if (avcodec_open2(m_ffmpeg.context, m_ffmpeg.codec, nullptr) < 0) {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }

    m_ffmpeg.frame_yuv = av_frame_alloc();
    m_ffmpeg.frame_bgr = av_frame_alloc();
    if (!m_ffmpeg.frame_yuv || !m_ffmpeg.frame_bgr) {
        fprintf(stderr, "Could not allocate video frames\n");
        exit(1);
    }
    
    m_ffmpeg.bgr_buffer = nullptr;
    m_ffmpeg.bgr_buffer_size = 0;

    // Create socket with default blocking mode for registration
    int sockfd;
    struct sockaddr_in server_address, from_addr;
    socklen_t from_len = sizeof(from_addr);
    char buffer[MAXLINE];

    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Set socket options for better performance
    int rcvbuf = 16 * 1024 * 1024; // 16MB receive buffer
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) < 0) {
        perror("setsockopt(SO_RCVBUF) failed");
    }

    // Bind socket
    memset(&server_address, 0, sizeof(server_address));
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(PORT);
    server_address.sin_addr.s_addr = INADDR_ANY;

    if (bind(sockfd, (const struct sockaddr*)&server_address, sizeof(server_address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    // Register with the server
    const char* registration = "Client registration";
    struct sockaddr_in server_dest_addr;
    memset(&server_dest_addr, 0, sizeof(server_dest_addr));
    server_dest_addr.sin_family = AF_INET;
    server_dest_addr.sin_port = htons(CLIENT_PORT);
    server_dest_addr.sin_addr.s_addr = inet_addr(SERVER_IP);

    sendto(sockfd, registration, strlen(registration), 0,
           (const struct sockaddr*)&server_dest_addr, sizeof(server_dest_addr));
    std::cout << "Client registration message sent." << std::endl;

    // Use a timeout for registration response
    struct timeval tv;
    tv.tv_sec = 5;  // 5 second timeout for registration
    tv.tv_usec = 0;
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        perror("setsockopt(SO_RCVTIMEO) failed");
    }

    int data = recvfrom(sockfd, buffer, MAXLINE, 0, (struct sockaddr*)&from_addr, &from_len);
    if (data < 0) {
        perror("recvfrom failed during registration");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    buffer[data] = '\0';
    std::cout << "Client: Received registration confirmation: " << buffer << std::endl;

    //Setting the socket to non-blocking mode for the main loop
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags == -1) {
        perror("fcntl F_GETFL failed");
        exit(EXIT_FAILURE);
    }
    if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) == -1) {
        perror("fcntl F_SETFL O_NONBLOCK failed");
        exit(EXIT_FAILURE);
    }

    std::cout << "Client: Waiting for video..." << std::endl;

    // Frame reconstruction map - keeps track of partial frames by timestamp
    std::map<uint32_t, FrameData> frame_map;
    uint32_t current_timestamp = 0;
    std::queue<std::vector<uint8_t>> complete_frames;
    static int frame_counter = 0;

    while (true) {
        // Process frames that are already complete first
        while (!complete_frames.empty()) {
            std::vector<uint8_t>& h264_buffer = complete_frames.front();
            
            // Create packet for decoding
            AVPacket* packet = av_packet_alloc();
            if (!packet) {
                std::cerr << "Failed to allocate packet" << std::endl;
                complete_frames.pop();
                continue;
            }
            
            packet->data = h264_buffer.data();
            packet->size = h264_buffer.size();

            int send_result = avcodec_send_packet(m_ffmpeg.context, packet);
            if (send_result < 0) {
                std::cerr << "Error sending packet: " << send_result << std::endl;
                // Try to recover from errors
                if (send_result == AVERROR_INVALIDDATA) {
                    avcodec_flush_buffers(m_ffmpeg.context);
                }
                av_packet_free(&packet);
                complete_frames.pop();
                continue;
            }

            bool frame_decoded = false;
            while (avcodec_receive_frame(m_ffmpeg.context, m_ffmpeg.frame_yuv) == 0) {
                frame_decoded = true;
                
                // Initialize conversion context if needed
                if (!m_ffmpeg.sws_ctx) {
                    m_ffmpeg.sws_ctx = sws_getContext(
                        m_ffmpeg.context->width, m_ffmpeg.context->height, m_ffmpeg.context->pix_fmt,
                        m_ffmpeg.context->width, m_ffmpeg.context->height, AV_PIX_FMT_BGR24,
                        SWS_BILINEAR, nullptr, nullptr, nullptr);
                }

                // Allocate BGR buffer just once and reuse
                int required_size = av_image_get_buffer_size(
                    AV_PIX_FMT_BGR24, m_ffmpeg.context->width, m_ffmpeg.context->height, 1);
                    
                if (required_size > m_ffmpeg.bgr_buffer_size) {
                    if (m_ffmpeg.bgr_buffer) {
                        av_free(m_ffmpeg.bgr_buffer);
                    }
                    m_ffmpeg.bgr_buffer = (uint8_t*)av_malloc(required_size * sizeof(uint8_t));
                    m_ffmpeg.bgr_buffer_size = required_size;
                    
                    // Set up BGR frame with the buffer
                    av_image_fill_arrays(
                        m_ffmpeg.frame_bgr->data, m_ffmpeg.frame_bgr->linesize,
                        m_ffmpeg.bgr_buffer, AV_PIX_FMT_BGR24,
                        m_ffmpeg.context->width, m_ffmpeg.context->height, 1);
                }

                // Convert YUV to BGR
                sws_scale(
                    m_ffmpeg.sws_ctx,
                    m_ffmpeg.frame_yuv->data, m_ffmpeg.frame_yuv->linesize,
                    0, m_ffmpeg.context->height,
                    m_ffmpeg.frame_bgr->data, m_ffmpeg.frame_bgr->linesize
                );

                // Create OpenCV Mat (using existing buffer, no deep copy needed)
                cv::Mat frame(
                    m_ffmpeg.context->height,
                    m_ffmpeg.context->width,
                    CV_8UC3,
                    m_ffmpeg.frame_bgr->data[0],
                    m_ffmpeg.frame_bgr->linesize[0]
                );

                // Calculate FPS
                auto current_time = std::chrono::high_resolution_clock::now();
                int64_t time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   current_time.time_since_epoch()).count();

                if (prev_frame_time > 0) {
                    // Calculate instantaneous FPS
                    double frame_time = (time_ns - prev_frame_time) / 1000000000.0;
                    double instant_fps = 1.0 / frame_time;
                    fps = instant_fps;

                }
                prev_frame_time = time_ns;

                // Overlay FPS on the frame
                std::stringstream fps_text;
                fps_text << "FPS: " << std::fixed << std::setprecision(1) << fps;
                cv::putText(frame, fps_text.str(), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

                // Display the frame
                cv::imshow("Video", frame);
                int key = cv::waitKey(1);
                if (key == 27) { // Exit on 'ESC' key
                    // Clean up and exit
                    if (m_ffmpeg.bgr_buffer) {
                        av_free(m_ffmpeg.bgr_buffer);
                    }
                    av_frame_free(&m_ffmpeg.frame_yuv);
                    av_frame_free(&m_ffmpeg.frame_bgr);
                    avcodec_free_context(&m_ffmpeg.context);
                    if (m_ffmpeg.sws_ctx) {
                        sws_freeContext(m_ffmpeg.sws_ctx);
                    }
                    close(sockfd);
                    av_packet_free(&packet);
                    return 0;
                }
            }
            
            av_packet_free(&packet);
            complete_frames.pop();
        }
        
        // Using select to efficiently wait for data
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(sockfd, &readfds);
        
        // Set a short timeout for select
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 1000; // 1ms timeout - more aggressive polling

        int activity = select(sockfd + 1, &readfds, NULL, NULL, &tv);
        
        if (activity < 0) {
            if (errno != EINTR) { // Ignore if interrupted by signal
                perror("select failed");
                break;
            }
            continue;
        }
        
        // No data available yet, continue processing or rendering
        if (activity == 0) {
            // Could add frame interpolation or other processing here
            continue;
        }
        
        // Data is available to read
        if (FD_ISSET(sockfd, &readfds)) {
            auto now = std::chrono::system_clock::now();
            uint64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                now.time_since_epoch()).count();
            std::time_t time_tt = std::chrono::system_clock::to_time_t(now);
            std::tm *tm_ptr = std::localtime(&time_tt);
            char time_str[50];
            std::strftime(time_str, sizeof(time_str), "%H:%M:%S", tm_ptr);
            
            // Batch receive - try to get multiple packets at once
            int consecutive_packets = 0;
            const int MAX_CONSECUTIVE_PACKETS = 10; // Limit to prevent starvation

            while (consecutive_packets < MAX_CONSECUTIVE_PACKETS) {
                data = recvfrom(sockfd, buffer, MAXLINE, 0, (struct sockaddr*)&from_addr, &from_len);
                
                if (data < 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        // No more data available right now
                        break;
                    } else {
                        perror("recvfrom failed");
                        break;
                    }
                }
                
                // Check if this is a header packet (8 bytes with frame info)
                if (data == 8) {
                    uint32_t total_size = 
                        ((uint8_t)buffer[0] << 24) | 
                        ((uint8_t)buffer[1] << 16) | 
                        ((uint8_t)buffer[2] << 8) | 
                        (uint8_t)buffer[3];
                        
                    uint32_t timestamp = 
                        ((uint8_t)buffer[4] << 24) | 
                        ((uint8_t)buffer[5] << 16) | 
                        ((uint8_t)buffer[6] << 8) | 
                        (uint8_t)buffer[7];
                    
                    // Initialize a new frame entry
                    frame_map[timestamp] = {
                        .data = std::vector<uint8_t>(),
                        .expected_size = total_size,
                        .timestamp = timestamp,
                        .complete = false
                    };
                    
                    frame_map[timestamp].data.reserve(total_size);
                    current_timestamp = timestamp;
                    
                    // Start frame reconstruction
                    continue;
                }
                
                // If we have a current timestamp, add data to that frame
                if (current_timestamp > 0 && frame_map.find(current_timestamp) != frame_map.end()) {
                    auto& frame = frame_map[current_timestamp];
                    
                    // Add received data to the frame buffer
                    frame.data.insert(frame.data.end(), buffer, buffer + data);
                    
                    // Check if frame is complete
                    if (frame.data.size() >= frame.expected_size) {
                        frame.complete = true;
                        
                        // Move to complete frames queue
                        complete_frames.push(frame.data);
                        
                        // Remove from map to free memory
                        frame_map.erase(current_timestamp);
                        current_timestamp = 0;
                    }
                }

                consecutive_packets++;
            }
        }
        
        // Clean up old incomplete frames (after 5 seconds)
        for (auto it = frame_map.begin(); it != frame_map.end();) {
            if (current_timestamp - it->first > 150) { // Assuming ~30fps, 5 seconds = 150 frames
                it = frame_map.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Cleanup
    if (m_ffmpeg.bgr_buffer) {
        av_free(m_ffmpeg.bgr_buffer);
    }
    av_frame_free(&m_ffmpeg.frame_yuv);
    av_frame_free(&m_ffmpeg.frame_bgr);
    avcodec_free_context(&m_ffmpeg.context);
    if (m_ffmpeg.sws_ctx) {
        sws_freeContext(m_ffmpeg.sws_ctx);
    }
    close(sockfd);

    return 0;
}