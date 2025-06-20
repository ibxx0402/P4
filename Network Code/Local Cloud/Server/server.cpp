// Server side implementation of UDP client-server model 
#include <opencv2/photo/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <algorithm>
#include <ctime>
#include <vector>
#include <queue>
#include <fstream>


// Include FFmpeg headers
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h> // For sws_getContext and SWS_BILINEAR
#include <libavutil/imgutils.h> // For av_image_get_buffer_size and av_image_fill_arrays
}

#define SERVER_IP "192.168.0.112"
#define CLIENT_PORT 9998
#define CAMERA_PORT 9999
#define MAXLINE 1024

const size_t MAX_PACKET_SIZE = 1400; // Smaller than MAX_UDP_SIZE to avoid fragmentation

// Extend FFmpegContext struct
struct FFmpegContext {
    const AVCodec* codec;
    AVCodecContext* context;
    AVFrame* frame_yuv;
    AVFrame* frame_bgr;
    SwsContext* sws_ctx;

    // Encoding-specific fields
    const AVCodec* encoder_codec;
    AVCodecContext* encoder_context;
    AVFrame* frame_encoder;
    AVPacket* packet_encoder;
};

FFmpegContext m_ffmpeg;

std::vector<uint8_t> h264_buffer; // Buffer to accumulate H.264 data

bool isCompleteNalUnit(const std::vector<uint8_t>& buffer) {
    // Check if the buffer has at least one NAL unit
    // Look for start code pattern: 0x00 0x00 0x01 or 0x00 0x00 0x00 0x01
    size_t i = 0;
    int start_codes_found = 0;

    while (i < buffer.size() - 3) {
        if ((buffer[i] == 0 && buffer[i+1] == 0 && buffer[i+2] == 1) ||
            (i < buffer.size() - 4 && buffer[i] == 0 && buffer[i+1] == 0 && 
             buffer[i+2] == 0 && buffer[i+3] == 1)) {
            
            start_codes_found++;
            if (start_codes_found >= 2) {
                return true; // Found at least 2 start codes (one complete NAL unit)
            }
            
            // Skip ahead to avoid finding the same start code
            i += 3;
        } else {
            i++;
        }
    }
    
    return false; // Not enough start codes found
}

// Function to find start of NAL unit in buffer
size_t findNalStartCode(const std::vector<uint8_t>& buffer, size_t start_pos = 0) {
    for (size_t i = start_pos; i < buffer.size() - 3; i++) {
        // Check for 3-byte start code: 0x00 0x00 0x01
        if (buffer[i] == 0 && buffer[i+1] == 0 && buffer[i+2] == 1) {
            return i;
        }
        // Check for 4-byte start code: 0x00 0x00 0x00 0x01
        if (i < buffer.size() - 4 && buffer[i] == 0 && buffer[i+1] == 0 && 
            buffer[i+2] == 0 && buffer[i+3] == 1) {
            return i;
        }
    }
    return buffer.size(); // Not found
}

// Driver code 
int main() { 
    // Initialize libav used to decode H.264
    avformat_network_init();
    m_ffmpeg.codec = avcodec_find_decoder_by_name("h264_cuvid"); // or other HW decoders
    if (!m_ffmpeg.codec) {
        // Fall back to software decoder
        m_ffmpeg.codec = avcodec_find_decoder(AV_CODEC_ID_H264);

    }
    m_ffmpeg.context = avcodec_alloc_context3(m_ffmpeg.codec);
    if (!m_ffmpeg.context)
    {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    

    // Add error resilience flags
    m_ffmpeg.context->err_recognition = AV_EF_CAREFUL;
    m_ffmpeg.context->flags |= AV_CODEC_FLAG_LOW_DELAY;
    m_ffmpeg.context->flags2 |= AV_CODEC_FLAG2_CHUNKS;
    m_ffmpeg.context->thread_count = 4; // Use multiple threads for decoding
    m_ffmpeg.context->thread_type = FF_THREAD_SLICE;
    m_ffmpeg.context->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

    if (avcodec_open2(m_ffmpeg.context, m_ffmpeg.codec, nullptr) < 0)
    {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }

    // Allocate YUV frame
    m_ffmpeg.frame_yuv = av_frame_alloc();
    if (!m_ffmpeg.frame_yuv) 
    {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }

    // Allocate BGR frame
    m_ffmpeg.frame_bgr = av_frame_alloc();
    if (!m_ffmpeg.frame_bgr) {
        fprintf(stderr, "Could not allocate BGR frame\n");
        exit(1);
    }

    // Initialize encoder
    m_ffmpeg.encoder_codec = avcodec_find_encoder_by_name("h264_videotoolbox");
    if (!m_ffmpeg.encoder_codec) {
        std::cerr << "No suitable encoder found" << std::endl;
        exit(1);
    }
    

    m_ffmpeg.encoder_context = avcodec_alloc_context3(m_ffmpeg.encoder_codec);
    if (!m_ffmpeg.encoder_context) {
        std::cerr << "Could not allocate encoder context" << std::endl;
        exit(1);
    }


    // Set encoder parameters
    m_ffmpeg.encoder_context->bit_rate = 1000000; // Adjust bitrate as needed
    m_ffmpeg.encoder_context->width = 1280;       // Set valid width
    m_ffmpeg.encoder_context->height = 720;      // Set valid height
    m_ffmpeg.encoder_context->time_base = {1, 60}; // 60 fps
    m_ffmpeg.encoder_context->framerate = {60, 1};
    m_ffmpeg.encoder_context->gop_size = 60;      // Group of pictures size
    m_ffmpeg.encoder_context->max_b_frames = 0;   // Disable B-frames for low latency
    m_ffmpeg.encoder_context->pix_fmt = AV_PIX_FMT_YUV420P; // Use YUV420P pixel format
    m_ffmpeg.encoder_context->refs = 2;          // Fewer reference frames = faster
    //m_ffmpeg.encoder_context->thread_count = 16; // More reasonable value
    m_ffmpeg.encoder_context->thread_type = FF_THREAD_FRAME; // Use frame-level threading


    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "preset", "ultrafast", 0); // Set encoding preset
    av_dict_set(&opts, "tune", "zerolatency", 0); // Set tuning option

    if (avcodec_open2(m_ffmpeg.encoder_context, m_ffmpeg.encoder_codec, &opts) < 0) {
        std::cerr << "Could not open encoder" << std::endl;
        exit(1);
    }

    av_dict_free(&opts); // Free the dictionary
   
    std::cout << "Using encoder: " << m_ffmpeg.encoder_codec->name << std::endl;

    // Allocate frame for encoding
    m_ffmpeg.frame_encoder = av_frame_alloc();
    if (!m_ffmpeg.frame_encoder) {
        std::cerr << "Could not allocate encoder frame" << std::endl;
        exit(1);
    }
    m_ffmpeg.frame_encoder->format = m_ffmpeg.encoder_context->pix_fmt;
    m_ffmpeg.frame_encoder->width = m_ffmpeg.encoder_context->width;
    m_ffmpeg.frame_encoder->height = m_ffmpeg.encoder_context->height;

    if (av_frame_get_buffer(m_ffmpeg.frame_encoder, 32) < 0) {
        std::cerr << "Could not allocate frame buffer for encoder" << std::endl;
        exit(1);
    }

    // Allocate packet for encoded data
    m_ffmpeg.packet_encoder = av_packet_alloc();
    if (!m_ffmpeg.packet_encoder) {
        std::cerr << "Could not allocate encoder packet" << std::endl;
        exit(1);
    }

    char buffer[MAXLINE]; 

    //Creating client socket
    //###################################################################################
    int client_sock; 
    struct sockaddr_in client_bind_addr, client_peer_addr; 
    
    // Creating socket file descriptor 
    if ( (client_sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } 
    
    memset(&client_bind_addr, 0, sizeof(client_bind_addr)); 
    memset(&client_peer_addr, 0, sizeof(client_peer_addr)); 
    
    // Filling server information for client
    client_bind_addr.sin_family = AF_INET; // IPv4 
    client_bind_addr.sin_addr.s_addr =  inet_addr(SERVER_IP); 
    client_bind_addr.sin_port = htons(CLIENT_PORT); 
    
    // Bind the socket with the server address 
    if ( bind(client_sock, (const struct sockaddr *)&client_bind_addr, 
            sizeof(client_bind_addr)) < 0 ) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 
    
    //##################################################################################

    //Creating server socket
    //##################################################################################
    int camera_sock; 
    struct sockaddr_in camera_addr, camera_peer_addr; 
    
    // Creating socket file descriptor 
    if ( (camera_sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } 
    
    memset(&camera_addr, 0, sizeof(camera_addr)); 
    memset(&camera_peer_addr, 0, sizeof(camera_peer_addr)); 
    
    // Filling server information for client
    camera_addr.sin_family = AF_INET; // IPv4 
    camera_addr.sin_addr.s_addr =  inet_addr(SERVER_IP); 
    camera_addr.sin_port = htons(CAMERA_PORT); 

    // After creating camera_sock, add:
    int enable = 1;
    if (setsockopt(camera_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0) {
        perror("setsockopt(SO_REUSEADDR) failed");
    }

    // Set receive buffer size to be large enough for video frames
    int rcvbuf = 64 * 1024 * 1024; // 64MB buffer
    int sndbuf = 64 * 1024 * 1024; // 64MB send buffer


    if (setsockopt(camera_sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) < 0) {
        perror("setsockopt(SO_RCVBUF) failed");
    }
    
    if (setsockopt(camera_sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf)) < 0) {
        perror("setsockopt(SO_SNDBUF) failed");
    }

    // Bind the socket with the server address
    if (bind(camera_sock, (const struct sockaddr *)&camera_addr,
            sizeof(camera_addr)) < 0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    //##################################################################################
    std::cout<<"Server: Listening for client registration on "<< SERVER_IP << ":" << CLIENT_PORT << " & " << CAMERA_PORT <<std::endl; 
    
    // Add this above your main loop:
    struct sockaddr_in registered_client_addr;
    socklen_t registered_client_len = 0;
    bool client_registered = false;

    while (true) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(client_sock, &readfds);
        FD_SET(camera_sock, &readfds);

        int maxfd = std::max(client_sock, camera_sock);

        int activity = select(maxfd + 1, &readfds, NULL, NULL, NULL); // Block forever

        if (activity > 0) {
            if (FD_ISSET(client_sock, &readfds)) {
                // Handle client_sock activity
                char buffer[1024];
                struct sockaddr_in from_addr;
                socklen_t from_len = sizeof(from_addr);

                ssize_t n = recvfrom(client_sock, buffer, sizeof(buffer), 0,
                                     (struct sockaddr *)&from_addr, &from_len);

                if (n > 0) {
                    buffer[n] = '\0'; // Null-terminate if you expect string data
                    std::cout << "Server: Received client message from "
                              << inet_ntoa(from_addr.sin_addr) << ":" << ntohs(from_addr.sin_port)
                              << " : " << buffer << std::endl;

                     // Store the client address for forwarding video
                    registered_client_addr = from_addr;
                    registered_client_len = from_len;
                    client_registered = true;

                    // After printing the client message
                    const char* confirmation = "Registration successful";
                    sendto(client_sock, confirmation, strlen(confirmation), 0, (struct sockaddr *)&from_addr, from_len);
                }
                
            }
            if (FD_ISSET(camera_sock, &readfds)) {
                static unsigned long long total_bytes = 0;
                static int packets = 0;
                static time_t last_stats = time(nullptr);
                
                char buffer[65536];
                struct sockaddr_in from_addr; 
                socklen_t from_len = sizeof(from_addr);
                
                int data = recvfrom(camera_sock, buffer, sizeof(buffer), 0, 
                                    (struct sockaddr *)&from_addr, &from_len);
                                    
                if (data > 0 && client_registered) {
                    //std::cout << "Server: Received " << data << " bytes from camera" << std::endl;
                    // Extend our H.264 buffer with new data
                    h264_buffer.insert(h264_buffer.end(), buffer, buffer + data);


                    while (isCompleteNalUnit(h264_buffer)) {
                        size_t nal_start = findNalStartCode(h264_buffer);
                        size_t next_nal_start = findNalStartCode(h264_buffer, nal_start + 3);
                        
                        if (next_nal_start > nal_start && next_nal_start < h264_buffer.size()) {
                            // Extract complete NAL unit
                            std::vector<uint8_t> nal_unit(h264_buffer.begin() + nal_start, h264_buffer.begin() + next_nal_start);
                            
                            // Create packet properly
                            AVPacket *packet = av_packet_alloc();
                            if (!packet) {
                                std::cerr << "Failed to allocate packet" << std::endl;
                                continue;
                            }
                            
                            // Allocate and copy data properly
                            uint8_t* packet_data = (uint8_t*)av_malloc(nal_unit.size());
                            if (!packet_data) {
                                std::cerr << "Failed to allocate packet data buffer" << std::endl;
                                av_packet_free(&packet);
                                continue;
                            }
                            
                            // Copy NAL unit to packet data buffer
                            memcpy(packet_data, nal_unit.data(), nal_unit.size());
                            
                            // Set up packet with proper buffer management
                            av_packet_from_data(packet, packet_data, nal_unit.size());
                            packet->pts = AV_NOPTS_VALUE;
                            packet->dts = AV_NOPTS_VALUE;

                            // Send the packet to the decoder
                            int send_result = avcodec_send_packet(m_ffmpeg.context, packet);
                            //char time_str3[50];
                            if (send_result == 0) {
                                // Try to receive decoded frame
                                int receive_result = avcodec_receive_frame(m_ffmpeg.context, m_ffmpeg.frame_yuv);
                                
                                if (receive_result == 0) {
                                    
									// Successfully decoded a frame
                                    if (!m_ffmpeg.sws_ctx) {
                                        m_ffmpeg.sws_ctx = sws_getContext(
                                            m_ffmpeg.context->width, m_ffmpeg.context->height, m_ffmpeg.context->pix_fmt,
                                            m_ffmpeg.context->width, m_ffmpeg.context->height, AV_PIX_FMT_BGR24,
                                            SWS_BILINEAR, nullptr, nullptr, nullptr);
                                    }

                                    // Properly initialize frame_bgr if not already done
                                    if (!m_ffmpeg.frame_bgr->width || !m_ffmpeg.frame_bgr->height) {
                                        m_ffmpeg.frame_bgr->format = AV_PIX_FMT_BGR24;
                                        m_ffmpeg.frame_bgr->width = m_ffmpeg.context->width;
                                        m_ffmpeg.frame_bgr->height = m_ffmpeg.context->height;
                                        
                                        // Allocate proper buffers for the frame
                                        if (av_frame_get_buffer(m_ffmpeg.frame_bgr, 32) < 0) {
                                            std::cerr << "Could not allocate BGR frame buffers" << std::endl;
                                            // Handle error
                                        }
                                    }

                                    // Make frame data writable
                                    if (av_frame_make_writable(m_ffmpeg.frame_bgr) < 0) {
                                        std::cerr << "Could not make BGR frame writable" << std::endl;
                                        // Handle error
                                    }

                                    // Now do the conversion
                                    sws_scale(
                                        m_ffmpeg.sws_ctx,
                                        m_ffmpeg.frame_yuv->data, m_ffmpeg.frame_yuv->linesize,
                                        0, m_ffmpeg.context->height,
                                        m_ffmpeg.frame_bgr->data, m_ffmpeg.frame_bgr->linesize
                                    );

                                    // Create OpenCV Mat that references the FFmpeg frame data
                                    cv::Mat frame(m_ffmpeg.context->height, 
                                                 m_ffmpeg.context->width, 
                                                 CV_8UC3, 
                                                 m_ffmpeg.frame_bgr->data[0], 
                                                 m_ffmpeg.frame_bgr->linesize[0]);

                                    // Create destination Mat for filtered result
                                    cv::Mat dst;
                                    // Apply OpenCV denoisiing 
                                    cv::bilateralFilter(frame, dst, 8, 10, 2);


                                    // Convert the frame to a GpuMat
                                    //cv::cuda::GpuMat gpu_frame, gpu_dst;
                                    //gpu_frame.upload(frame);

                                    // Apply CUDA-based denoising
                                    //cv::cuda::fastNlMeansDenoisingColored(gpu_frame, gpu_dst, 2, 3, 7, 3);

                                    // Download the result back to a standard Mat
                                    //gpu_dst.download(dst); 

                                    // Copy the filtered data back to the FFmpeg frame
                                    memcpy(m_ffmpeg.frame_bgr->data[0], dst.data, dst.step * dst.rows);

                                    // Create a separate sws context for BGR to YUV conversion
                                    SwsContext* sws_ctx_encoder = sws_getContext(
                                        m_ffmpeg.encoder_context->width, m_ffmpeg.encoder_context->height, AV_PIX_FMT_BGR24,
                                        m_ffmpeg.encoder_context->width, m_ffmpeg.encoder_context->height, m_ffmpeg.encoder_context->pix_fmt,
                                        SWS_BILINEAR, nullptr, nullptr, nullptr
                                    );

                                    if (!sws_ctx_encoder) {
                                        std::cerr << "Could not initialize sws context for encoder" << std::endl;
                                        exit(1);
                                    }

                                    // Make encoder frame writable
                                    if (av_frame_make_writable(m_ffmpeg.frame_encoder) < 0) {
                                        std::cerr << "Could not make encoder frame writable" << std::endl;
                                        // Handle error
                                    }

                                    // Convert BGR to YUV for encoding
                                    sws_scale(
                                        sws_ctx_encoder,
                                        m_ffmpeg.frame_bgr->data, m_ffmpeg.frame_bgr->linesize,
                                        0, m_ffmpeg.encoder_context->height,
                                        m_ffmpeg.frame_encoder->data, m_ffmpeg.frame_encoder->linesize
                                    );

                                    // Free the temporary sws context
                                    sws_freeContext(sws_ctx_encoder);

                                    // Set frame PTS (presentation timestamp)
                                    m_ffmpeg.frame_encoder->pts = av_rescale_q(packets, m_ffmpeg.encoder_context->time_base, m_ffmpeg.encoder_context->time_base);
                                    packets++;

                                    // Send the frame to the encoder
                                    int ret = avcodec_send_frame(m_ffmpeg.encoder_context, m_ffmpeg.frame_encoder);
                                    if (ret < 0) {
                                        std::cerr << "Error sending frame for encoding" << std::endl;
                                    } else {
                                        // Get the encoded packets
                                        while (ret >= 0) {
                                            ret = avcodec_receive_packet(m_ffmpeg.encoder_context, m_ffmpeg.packet_encoder);
                                            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                                                // Need more input or end of stream
                                                break;
                                            } else if (ret < 0) {
                                                std::cerr << "Error during encoding" << std::endl;
                                                break;
                                            }

                                            // Successfully got an encoded packet, send it to the client
                                            if (client_registered) {
                                                // Send the encoded packet to the registered client
                                                size_t packet_size = m_ffmpeg.packet_encoder->size;
                                                
                                                // Split into smaller packets if needed (to avoid UDP fragmentation)
                                                
                                                // Send header with information about the packet
                                                uint8_t header[8];
                                                // First 4 bytes: total size of encoded frame
                                                header[0] = (packet_size >> 24) & 0xFF;
                                                header[1] = (packet_size >> 16) & 0xFF;
                                                header[2] = (packet_size >> 8) & 0xFF;
                                                header[3] = packet_size & 0xFF;
                                                // Next 4 bytes: packet timestamp
                                                uint32_t timestamp = packets; // Use packet counter as timestamp
                                                header[4] = (timestamp >> 24) & 0xFF;
                                                header[5] = (timestamp >> 16) & 0xFF;
                                                header[6] = (timestamp >> 8) & 0xFF;
                                                header[7] = timestamp & 0xFF;
                                                
                                                // Send header
                                                sendto(client_sock, header, sizeof(header), 0, 
                                                       (struct sockaddr *)&registered_client_addr, registered_client_len);
                                                
                                                // Send data in chunks
                                                for (size_t offset = 0; offset < packet_size; offset += MAX_PACKET_SIZE) {
                                                    size_t chunk_size = std::min(MAX_PACKET_SIZE, packet_size - offset);
                                                    sendto(client_sock, m_ffmpeg.packet_encoder->data + offset, chunk_size, 0, 
                                                           (struct sockaddr *)&registered_client_addr, registered_client_len);
                                                    
                                                    // Small delay to prevent overwhelming the network or receiver
                                                    //usleep(1000); // 1000us delay between chunks
                                                }
                                            }
                                            
                                            // Unref the packet for reuse
                                            av_packet_unref(m_ffmpeg.packet_encoder);
                                        }
                                    }
                                } 
                                else {
                                    char err_buf[AV_ERROR_MAX_STRING_SIZE];
                                    std::cerr << "Error decoding frame: " 
                                              << av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, receive_result) 
                                              << std::endl;
                                    if (receive_result == AVERROR(EAGAIN)) {
                                        std::cout << "Decoder needs more data" << std::endl;
                                    } else if (receive_result == AVERROR_EOF) {
                                        std::cout << "End of stream reached" << std::endl;
                                    } else {
                                        // Reset the decoder after serious errors
                                        avcodec_flush_buffers(m_ffmpeg.context);
                                    }
                                }
                                
                                // Unref the frame to prepare for next decode
                                av_frame_unref(m_ffmpeg.frame_yuv);
                            }
                            // Free the packet
                            av_packet_free(&packet);
                            
                            // Remove processed NAL unit from buffer
                            h264_buffer.erase(h264_buffer.begin(), h264_buffer.begin() + next_nal_start);
                        } else {
                            // Not enough data for a complete NAL unit
                            break;
                        }
                    }
       
                    // If buffer gets too large, trim it
                    if (h264_buffer.size() > 1000000) { // 1MB threshold
                        size_t nal_start = findNalStartCode(h264_buffer);
                        if (nal_start < h264_buffer.size()) {
                            h264_buffer.erase(h264_buffer.begin(), h264_buffer.begin() + nal_start);
                        } else {
                            h264_buffer.erase(h264_buffer.begin(), h264_buffer.begin() + h264_buffer.size() / 2);
                        }
                        std::cout << "Buffer trimmed to " << h264_buffer.size() << " bytes" << std::endl;
                    }
                }
            }
        }
    }

    // Free FFmpeg resources
    if (m_ffmpeg.frame_yuv) {
        av_frame_free(&m_ffmpeg.frame_yuv);
    }
    if (m_ffmpeg.frame_bgr) {
        av_frame_free(&m_ffmpeg.frame_bgr);
    }
    if (m_ffmpeg.context) {
        avcodec_free_context(&m_ffmpeg.context);
    }
    if (m_ffmpeg.frame_encoder) {
        av_frame_free(&m_ffmpeg.frame_encoder);
    }
    if (m_ffmpeg.packet_encoder) {
        av_packet_free(&m_ffmpeg.packet_encoder);
    }
    if (m_ffmpeg.encoder_context) {
        avcodec_free_context(&m_ffmpeg.encoder_context);
    }

    // Close sockets
    close(client_sock);
    close(camera_sock);

    return 0; 
}
