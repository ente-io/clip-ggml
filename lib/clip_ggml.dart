import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:math';
import 'package:ffi/ffi.dart';

typedef load_model_request = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> model_path);

class CLIP {
  static const clipLib = "libclip_ggml.so";

  static final _clip = openLib();

  static ffi.DynamicLibrary openLib() {
    if (Platform.isIOS || Platform.isMacOS) {
      return ffi.DynamicLibrary.process();
    } else {
      return ffi.DynamicLibrary.open(clipLib);
    }
  }

  static loadImageModel(String modelPath) {
    var res = _clip
        .lookupFunction<load_model_request, load_model_request>(
            "load_image_model")
        .call(modelPath.toNativeUtf8());
    return res.toDartString();
  }
}
