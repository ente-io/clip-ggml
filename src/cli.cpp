#include <string>
#include <vector>
#include <thread>

struct cli_params {
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

    std::string img_model;
    std::string txt_model;
    std::string image_path;
    std::string text;
    int verbose = 1;
};

void print_help(int argc, char ** argv, cli_params & params) {
    printf("Usage: %s [options]\n", argv[0]);
    printf("\nOptions:");
    printf("  -h, --help: Show this message and exit\n");
    printf("  -mv <path>, --modelvision <path>: path to vision model. Default: %s\n", params.img_model.c_str());
    printf("  -mt <path>, --modeltext <path>: path to text model. Default: %s\n", params.txt_model.c_str());
    printf("  -t N, --threads N: Number of threads to use for inference. Default: %d\n", params.n_threads);
    printf("  --text <text>: Text to encode.\n");
    printf("  --image <path>: Path to an image file.\n");
    printf("  -v <level>, --verbose <level>: Control the level of verbosity. 0 = minimum, 2 = maximum. Default: %d\n",
           params.verbose);
}

bool cli_params_parse(int argc, char ** argv, cli_params & params) {
    for (int i = 0; i < argc; i++) {
        std::string arg = std::string(argv[i]);
        if (arg == "--modelvision") {
            params.img_model = argv[++i];
        } else if (arg == "--modeltext") {
            params.txt_model = argv[++i];
        } else if (arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "--text") {
            params.text = argv[++i];
        } else if (arg == "--image") {
            params.image_path = argv[++i];
        } else if (arg == "--verbose") {
            params.verbose = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            print_help(argc, argv, params);
            exit(0);
        } else {
            if (i != 0) {
                printf("%s: unrecognized argument: %s\n", __func__, arg.c_str());
                return false;
            }
        }
    }

    return !(params.image_path.empty() && params.text.empty());
}
