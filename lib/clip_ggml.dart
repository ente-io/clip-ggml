import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:ffi/ffi.dart';

typedef inference_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> image_path, ffi.Pointer<Utf8> text);

typedef load_model_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> model_path);

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

  String runInference({
    required String imagePath,
    required String text,
  }) {
    ffi.Pointer<Utf8> image_string = imagePath.toNativeUtf8();
    ffi.Pointer<Utf8> text_string = text.toNativeUtf8();
    var res = openLib()
        .lookupFunction<inference_request, inference_request>("run_inference")
        .call(image_string, text_string);
    return res.toDartString();
  }
}
