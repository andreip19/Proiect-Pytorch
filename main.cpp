#include <torch/script.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstdint>
#include <exception>
#include <string>
// test
namespace fs = std::filesystem;

int main() {
    try {
        const fs::path model_path = "C:/Users/User/Desktop/Proiect-Pytorch/model_traced.pt";
        const fs::path bin_dir = "C:/Users/User/Desktop/Proiect-Pytorch/bin_images";

    
        if (!fs::exists(model_path)) {
            std::cerr << "Eroare: model_traced.pt nu exista!" << std::endl;
            return 1;
        }

        std::ifstream model_stream(model_path, std::ios::binary);
        if (!model_stream.is_open()) {
            std::cerr << "Eroare: nu pot deschide modelul." << std::endl;
            return 1;
        }

        auto module = torch::jit::load(model_stream, torch::kCPU);
        module.eval();
        std::cout << "Modelul a fost incarcat cu succes!\n" << std::endl;

        if (!fs::exists(bin_dir) || !fs::is_directory(bin_dir)) {
            std::cerr << "Eroare: directorul " << bin_dir.string() << " nu exista!" << std::endl;
            std::cerr << "Ruleaza mai intai prepare_image.py pentru a genera fisierele .bin." << std::endl;
            return 1;
        }

        constexpr std::size_t IMAGE_SIZE = 28 * 28;

     
        for (const auto& entry : fs::directory_iterator(bin_dir)) {
            if (entry.path().extension() != ".bin") {
                continue;
            }

            std::ifstream image_stream(entry.path(), std::ios::binary);
            if (!image_stream.is_open()) {
                std::cerr << "Eroare la deschiderea fisierului: " << entry.path().filename().string() << std::endl;
                continue;
            }

            std::vector<std::uint8_t> pixels(IMAGE_SIZE);
            image_stream.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));

            if (image_stream.gcount() != static_cast<std::streamsize>(IMAGE_SIZE)) {
                std::cerr << "Eroare dimensiune (ignorare): " << entry.path().filename().string() << std::endl;
                continue;
            }

            std::vector<float> normalized_pixels(IMAGE_SIZE);
            for (std::size_t i = 0; i < IMAGE_SIZE; ++i) {
                normalized_pixels[i] = static_cast<float>(pixels[i]) / 255.0f;
            }

            auto input = torch::from_blob(
                normalized_pixels.data(),
                {1, 1, 28, 28},
                torch::TensorOptions().dtype(torch::kFloat32)
            ).clone();

            torch::NoGradGuard no_grad;
            auto output = module.forward({input});
            
            if (!output.isTensor()) {
                std::cerr << "Eroare: rezultatul nu este un tensor pentru " << entry.path().filename().string() << std::endl;
                continue;
            }

            auto result = output.toTensor();
            auto prediction = result.argmax(1);

            std::cout << "File: " << entry.path().filename().string() 
                      << " -> Predicted: " << prediction.item<int>() << std::endl;
        }
        
        std::cout << "\nPROCESAREA FOLDERULUI S-A INCHEIAT CU SUCCES!" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Eroare LibTorch:\n" << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Eroare C++:\n" << e.what() << std::endl;
        return 1;
    }

    return 0;
}