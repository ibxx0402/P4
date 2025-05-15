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
#include <opencv2/opencv.hpp>

// FFmpeg includes
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#define SERVER_IP "192.168.0.125"
#define PORT 9995
#define CLIENT_PORT 9998
#define MAXLINE 65507 // Max UDP packet size

struct FFmpegContext {
    const AVCodec* codec;
    AVCodecContext* context;
    AVFrame* frame_yuv;
    AVFrame* frame_bgr;
    SwsContext* sws_ctx;
};

FFmpegContext m_ffmpeg;

int main() {
    // Initialize FFmpeg
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

    // Create socket
    int sockfd;
    struct sockaddr_in server_address, from_addr;
    socklen_t from_len = sizeof(from_addr);
    char buffer[MAXLINE];

    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

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

    int data = recvfrom(sockfd, buffer, MAXLINE, 0, (struct sockaddr*)&from_addr, &from_len);
    buffer[data] = '\0';
    std::cout << "Client: Received registration confirmation: " << buffer << std::endl;

    std::cout << "Client: Waiting for video..." << std::endl;

    std::vector<uint8_t> h264_buffer;

    while (true) {
        // Receive video data
        data = recvfrom(sockfd, buffer, MAXLINE, 0, (struct sockaddr*)&from_addr, &from_len);
        if (data < 0) {
            perror("recvfrom failed");
            continue;
        }

        // Append received data to the buffer
        h264_buffer.insert(h264_buffer.end(), buffer, buffer + data);

        // Decode the video packet
        AVPacket* packet = av_packet_alloc();
        packet->data = h264_buffer.data();
        packet->size = h264_buffer.size();

        if (avcodec_send_packet(m_ffmpeg.context, packet) == 0) {
            while (avcodec_receive_frame(m_ffmpeg.context, m_ffmpeg.frame_yuv) == 0) {
                // Convert YUV to BGR for OpenCV
                if (!m_ffmpeg.sws_ctx) {
                    m_ffmpeg.sws_ctx = sws_getContext(
                        m_ffmpeg.context->width, m_ffmpeg.context->height, m_ffmpeg.context->pix_fmt,
                        m_ffmpeg.context->width, m_ffmpeg.context->height, AV_PIX_FMT_BGR24,
                        SWS_BILINEAR, nullptr, nullptr, nullptr);
                }

                int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, m_ffmpeg.context->width, m_ffmpeg.context->height, 1);
                uint8_t* bgr_buffer = (uint8_t*)av_malloc(num_bytes * sizeof(uint8_t));
                av_image_fill_arrays(m_ffmpeg.frame_bgr->data, m_ffmpeg.frame_bgr->linesize, bgr_buffer, AV_PIX_FMT_BGR24, m_ffmpeg.context->width, m_ffmpeg.context->height, 1);

                sws_scale(
                    m_ffmpeg.sws_ctx,
                    m_ffmpeg.frame_yuv->data, m_ffmpeg.frame_yuv->linesize,
                    0, m_ffmpeg.context->height,
                    m_ffmpeg.frame_bgr->data, m_ffmpeg.frame_bgr->linesize
                );

                // Create OpenCV Mat from the BGR frame
                cv::Mat frame(m_ffmpeg.context->height, m_ffmpeg.context->width, CV_8UC3, m_ffmpeg.frame_bgr->data[0], m_ffmpeg.frame_bgr->linesize[0]);

                // Display the frame
                cv::imshow("Video", frame);
                if (cv::waitKey(1) == 27) { // Exit on 'ESC' key
                    break;
                }

                av_free(bgr_buffer);
            }
        }

        av_packet_free(&packet);

        // Clear the buffer if it grows too large
        if (h264_buffer.size() > 500000) {
            h264_buffer.clear();
        }
    }

    // Cleanup
    av_frame_free(&m_ffmpeg.frame_yuv);
    av_frame_free(&m_ffmpeg.frame_bgr);
    avcodec_free_context(&m_ffmpeg.context);
    if (m_ffmpeg.sws_ctx) {
        sws_freeContext(m_ffmpeg.sws_ctx);
    }
    close(sockfd);

    return 0;
}