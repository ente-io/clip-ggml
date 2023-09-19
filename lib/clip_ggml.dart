import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:ffi/ffi.dart';

typedef load_model_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> model_path);

typedef create_image_embedding_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> image_path);

typedef inference_request = ffi.Pointer<Utf8> Function(ffi.Pointer<Utf8> text);

class CLIP {
  final String clipLib = "libclip_ggml.so";

  ffi.DynamicLibrary openLib() {
    if (Platform.isIOS || Platform.isMacOS) {
      return ffi.DynamicLibrary.process();
    } else {
      return ffi.DynamicLibrary.open(clipLib);
    }
  }

  String loadModel(String modelPath) {
    ffi.Pointer<Utf8> model_string = modelPath.toNativeUtf8();
    var res = openLib()
        .lookupFunction<load_model_request, load_model_request>("load_model")
        .call(model_string);
    return res.toDartString();
  }

  String createImageEmbedding(String imagePath) {
    ffi.Pointer<Utf8> image_string = imagePath.toNativeUtf8();
    var res = openLib()
        .lookupFunction<create_image_embedding_request,
            create_image_embedding_request>("create_image_embedding")
        .call(image_string);
    return res.toDartString();
  }

  String runInference(String text) {
    ffi.Pointer<Utf8> text_string = text.toNativeUtf8();
    var res = openLib()
        .lookupFunction<inference_request, inference_request>("run_inference")
        .call(text_string);
    return res.toDartString();
  }
}
