// UDP H.264 video receiver with improved error handling and naive reassembly using FFmpeg and OpenCV

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>

// FFmpeg includes
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/error.h>
#include <libswscale/swscale.h>
}

#define SERVER_IP "192.168.0.134"       // Local interface IP to bind - adjust as needed
#define VIDEO_PORT 9999                 // Port to listen on for video feed
#define MAXLINE 65507                   // Maximum UDP packet size

// Define a maximum size for the accumulated access unit (2 MB in this case)
#define MAX_FRAME_BUFFER_SIZE (2 * 1024 * 1024)
int frameCount = 0;
struct FFmpegContext {
    const AVCodec* codec;
    AVCodecContext* context;
    AVFrame* frame_yuv;
    AVFrame* frame_bgr;
    SwsContext* sws_ctx;
    uint8_t* bgr_buffer;  // Buffer for BGR conversion
    int bgr_buffer_size;
};

// Helper: check if data begins with an Annex B start code.
bool startsWithStartCode(const uint8_t* data, int size) {
    if (size >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1)
        return true;
    if (size >= 3 && data[0] == 0 && data[1] == 0 && data[2] == 1)
        return true;
    return false;
}

// Flush the currently accumulated frameBuffer as an access unit into the decoder.
void flushFrameBuffer(std::vector<uint8_t>& frameBuffer, FFmpegContext &m_ffmpeg) {
    if (frameBuffer.empty()) return;
    
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        std::cerr << "Failed to allocate packet" << std::endl;
        frameBuffer.clear();
        return;
    }

    int bufSize = frameBuffer.size();
    if (av_new_packet(packet, bufSize) < 0) {
        std::cerr << "Failed to allocate new packet" << std::endl;
        av_packet_free(&packet);
        frameBuffer.clear();
        return;
    }
    memcpy(packet->data, frameBuffer.data(), bufSize);

    int send_result = avcodec_send_packet(m_ffmpeg.context, packet);
    if (send_result < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
        av_strerror(send_result, errbuf, sizeof(errbuf));
        std::cerr << "Error sending packet: " << errbuf << std::endl;
        av_packet_free(&packet);
        // In case of a sending error, flush the decoder buffers to attempt recovery.
        avcodec_flush_buffers(m_ffmpeg.context);
        frameBuffer.clear();
        return;
    }
    static int frame_counter = 0;
    
    // Attempt to receive and process all available frames.
    while (true) {
        int ret = avcodec_receive_frame(m_ffmpeg.context, m_ffmpeg.frame_yuv);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break;
        else if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
            av_strerror(ret, errbuf, sizeof(errbuf));
            std::cerr << "Error during decoding: " << errbuf << ". Flushing decoder buffers." << std::endl;
            avcodec_flush_buffers(m_ffmpeg.context);
            break;
        }

        // Initialize conversion context if needed.
        if (!m_ffmpeg.sws_ctx) {
            m_ffmpeg.sws_ctx = sws_getContext(
                m_ffmpeg.context->width,
                m_ffmpeg.context->height,
                m_ffmpeg.context->pix_fmt,
                m_ffmpeg.context->width,
                m_ffmpeg.context->height,
                AV_PIX_FMT_BGR24,
                SWS_BILINEAR,
                nullptr, nullptr, nullptr);
            if (!m_ffmpeg.sws_ctx) {
                std::cerr << "Could not initialize SWScale context" << std::endl;
                break;
            }
        }

        // Allocate (or reallocate) the BGR buffer if needed.
        int required_size = av_image_get_buffer_size(AV_PIX_FMT_BGR24,
                                                     m_ffmpeg.context->width,
                                                     m_ffmpeg.context->height,
                                                     1);
        if (required_size > m_ffmpeg.bgr_buffer_size) {
            if (m_ffmpeg.bgr_buffer)
                av_free(m_ffmpeg.bgr_buffer);
            m_ffmpeg.bgr_buffer = (uint8_t*)av_malloc(required_size);
            if (!m_ffmpeg.bgr_buffer) {
                std::cerr << "Failed to allocate BGR buffer" << std::endl;
                break;
            }
            m_ffmpeg.bgr_buffer_size = required_size;
            av_image_fill_arrays(m_ffmpeg.frame_bgr->data, m_ffmpeg.frame_bgr->linesize,
                                 m_ffmpeg.bgr_buffer, AV_PIX_FMT_BGR24,
                                 m_ffmpeg.context->width, m_ffmpeg.context->height, 1);
        }

        // Convert YUV frame to BGR for OpenCV.
        sws_scale(m_ffmpeg.sws_ctx,
                  m_ffmpeg.frame_yuv->data, m_ffmpeg.frame_yuv->linesize,
                  0, m_ffmpeg.context->height,
                  m_ffmpeg.frame_bgr->data, m_ffmpeg.frame_bgr->linesize);

        // Wrap BGR data in an OpenCV Mat (no deep copy).
        cv::Mat frame(m_ffmpeg.context->height,
                      m_ffmpeg.context->width,
                      CV_8UC3,
                      m_ffmpeg.frame_bgr->data[0],
                      m_ffmpeg.frame_bgr->linesize[0]);

        // Register Timestamp
        int label = frame_counter++;
        auto now = std::chrono::system_clock::now();
        uint64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch()).count();
        std::time_t time_tt = std::chrono::system_clock::to_time_t(now);
        std::tm *tm_ptr = std::localtime(&time_tt);
        char time_str[50];
        std::strftime(time_str, sizeof(time_str), "%H:%M:%S", tm_ptr);

        // Log label and timestamp
        std::string log_path =  "log.txt";
        std::ofstream pi_log(log_path, std::ios::app);
        if (pi_log.is_open()) {
            // Log format: label time_string ts_ms sigma_before sigma_after
            pi_log << label << " " << ts_ms << "\n";
            pi_log.close();
        } else {
            std::cerr << "Unable to open log.txt for writing at path: " << log_path << "\n";
        }

        // Overlay text
        std::stringstream overlay_text;
        overlay_text << "Frame " << label << " " << time_str << " " << ts_ms;
        // Define the original position.
        cv::putText(frame, overlay_text.str(), cv::Point(10, frame.rows - 70),
        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255), 2);


        // Get the home directory
        std::string home_dir = getenv("HOME");
        //std::string frames_dir = home_dir + "/Documents/frames";
        // Construct the filename using frameCount (padded with zeros)
        std::ostringstream filename;
        filename << home_dir + "/Documents/frames/frame_" << std::setfill('0') << std::setw(5) << frameCount << ".jpg";
        // Save the frame using imwrite
        if (!imwrite(filename.str(), frame)) {
            std::cerr << "Error: Could not write image to " << filename.str() << std::endl;
        } else {
            std::cout << "Saved " << filename.str() << std::endl;
        }

        frameCount++;
        // Display the frame.
        cv::imshow("Video", frame);
        int key = cv::waitKey(1);
        if (key == 27) { // Exit if 'ESC' is pressed.
            av_packet_free(&packet);
            exit(0);
        }

        // Prepare for next frame.
        av_frame_unref(m_ffmpeg.frame_yuv);
    }
    
    av_packet_free(&packet);
    frameBuffer.clear();
}

int main() {
    // Optionally set FFmpeg logging level.
    av_log_set_level(AV_LOG_INFO);
    avformat_network_init();

    FFmpegContext m_ffmpeg = {};
    m_ffmpeg.codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!m_ffmpeg.codec) {
        std::cerr << "Codec not found" << std::endl;
        exit(1);
    }
    m_ffmpeg.context = avcodec_alloc_context3(m_ffmpeg.codec);
    if (!m_ffmpeg.context) {
        std::cerr << "Could not allocate video codec context" << std::endl;
        exit(1);
    }
    m_ffmpeg.context->err_recognition = AV_EF_CAREFUL;
    m_ffmpeg.context->flags |= AV_CODEC_FLAG_LOW_DELAY;
    m_ffmpeg.context->flags2 |= AV_CODEC_FLAG2_CHUNKS;
    if (avcodec_open2(m_ffmpeg.context, m_ffmpeg.codec, nullptr) < 0) {
        std::cerr << "Could not open codec" << std::endl;
        exit(1);
    }
    
    m_ffmpeg.frame_yuv = av_frame_alloc();
    m_ffmpeg.frame_bgr = av_frame_alloc();
    if (!m_ffmpeg.frame_yuv || !m_ffmpeg.frame_bgr) {
        std::cerr << "Could not allocate video frames" << std::endl;
        exit(1);
    }
    m_ffmpeg.bgr_buffer = nullptr;
    m_ffmpeg.bgr_buffer_size = 0;
    m_ffmpeg.sws_ctx = nullptr;
    
    // Create UDP socket.
    int sockfd;
    struct sockaddr_in local_address, from_addr;
    socklen_t from_len = sizeof(from_addr);
    char buffer[MAXLINE];
    
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }
    
    // Increase the receive buffer size (optional).
    int rcvbuf = 8 * 1024 * 1024; // 8 MB
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) < 0) {
        perror("setsockopt(SO_RCVBUF) failed");
    }
    
    // Set receive timeout (optional).
    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        perror("setsockopt(SO_RCVTIMEO) failed");
    }
    
    // Bind the socket to the designated IP and port.
    memset(&local_address, 0, sizeof(local_address));
    local_address.sin_family = AF_INET;
    local_address.sin_port = htons(VIDEO_PORT);
    local_address.sin_addr.s_addr = inet_addr(SERVER_IP);
    if (bind(sockfd, (struct sockaddr*)&local_address, sizeof(local_address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }
    
    std::cout << "Listening for video feed on " << SERVER_IP << ":" << VIDEO_PORT << std::endl;
    
    // Use a vector to accumulate UDP packet data that may be fragments of a frame.
    std::vector<uint8_t> frameBuffer;
    
    while (true) {
        int packet_size = recvfrom(sockfd, buffer, MAXLINE, 0, (struct sockaddr*)&from_addr, &from_len);
        if (packet_size < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                std::cout << "Receive timeout - no data available" << std::endl;
            else
                perror("recvfrom failed");
            continue;
        }
        
        bool hasStartCode = startsWithStartCode(reinterpret_cast<uint8_t*>(buffer), packet_size);

        // If a new start code is encountered and we've already built up a frame,
        // treat the accumulated data as one access unit.
        if (hasStartCode && !frameBuffer.empty()) {
            flushFrameBuffer(frameBuffer, m_ffmpeg);
        }
        
        // Append the current packet to the frame buffer.
        frameBuffer.insert(frameBuffer.end(), buffer, buffer + packet_size);
        
        // If the buffer grows too large (say, above MAX_FRAME_BUFFER_SIZE), flush it.
        if (frameBuffer.size() > MAX_FRAME_BUFFER_SIZE) {
            flushFrameBuffer(frameBuffer, m_ffmpeg);
        }
    }
    
    // Cleanup.
    if (m_ffmpeg.bgr_buffer)
        av_free(m_ffmpeg.bgr_buffer);
    av_frame_free(&m_ffmpeg.frame_yuv);
    av_frame_free(&m_ffmpeg.frame_bgr);
    avcodec_free_context(&m_ffmpeg.context);
    if (m_ffmpeg.sws_ctx)
        sws_freeContext(m_ffmpeg.sws_ctx);
    close(sockfd);
    
    return 0;
}
