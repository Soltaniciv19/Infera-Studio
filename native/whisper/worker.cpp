#include "whisper.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace {

bool read_size(std::uint32_t & size) {
    unsigned char bytes[4]{};
    if (!std::cin.read(reinterpret_cast<char *>(bytes), sizeof(bytes))) {
        return false;
    }
    size = std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8)
        | (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
    return true;
}

void write_size(std::uint32_t size) {
    const unsigned char bytes[] = {
        static_cast<unsigned char>(size),
        static_cast<unsigned char>(size >> 8),
        static_cast<unsigned char>(size >> 16),
        static_cast<unsigned char>(size >> 24),
    };
    std::cout.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
        std::cout << "infera-whisper-worker " << whisper_version() << '\n';
        return 0;
    }
    if (argc != 2) {
        return 1;
    }

    auto config = whisper_context_default_params();
    config.use_gpu = false;
    const std::unique_ptr<whisper_context, decltype(&whisper_free)> context(
        whisper_init_from_file_with_params(argv[1], config), whisper_free);
    if (!context) {
        return 2;
    }

    auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language = "auto";
    params.no_context = true;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.suppress_nst = true;
    params.token_timestamps = true;
    params.split_on_word = true;
    params.max_len = 72;
    params.n_threads = static_cast<int>(std::clamp(std::thread::hardware_concurrency(), 1u, 4u));

    std::cout.put('R').flush();
    std::uint32_t count = 0;
    while (read_size(count)) {
        if (count == 0 || count > 30 * WHISPER_SAMPLE_RATE) {
            return 3;
        }
        std::vector<unsigned char> pcm(count * 2);
        if (!std::cin.read(reinterpret_cast<char *>(pcm.data()), pcm.size())) {
            return 3;
        }
        std::vector<float> samples(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            const int value = pcm[2 * i] | (pcm[2 * i + 1] << 8);
            samples[i] = (value >= 32768 ? value - 65536 : value) / 32768.0f;
        }
        // One encoder position covers 20 ms. Leave two seconds of right padding.
        const int positions = (static_cast<int>(count) + 319) / 320 + 100;
        params.audio_ctx = std::min(whisper_n_audio_ctx(context.get()), std::max(256, ((positions + 63) / 64) * 64));
        if (whisper_full(context.get(), params, samples.data(), static_cast<int>(count)) != 0) {
            return 4;
        }

        const int segments = whisper_full_n_segments(context.get());
        write_size(segments);
        for (int i = 0; i < segments; ++i) {
            const char * text = whisper_full_get_segment_text(context.get(), i);
            const auto length = std::strlen(text);
            write_size(static_cast<std::uint32_t>(whisper_full_get_segment_t0(context.get(), i)));
            write_size(static_cast<std::uint32_t>(whisper_full_get_segment_t1(context.get(), i)));
            write_size(static_cast<std::uint32_t>(length));
            std::cout.write(text, length);
        }
        std::cout.flush();
    }

    return std::cin.eof() ? 0 : 3;
}
