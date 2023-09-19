import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:ffi/ffi.dart';

typedef clip_ggml_request_native = ffi.Pointer<Utf8> Function(
    ffi.Pointer<Utf8> dart_model_path,
    ffi.Pointer<Utf8> dart_image_path,
    ffi.Pointer<Utf8> dart_text);

class Clip_GGML {
  String clip_lib = "libclip_ggml.so";
  Clip_GGML({
    String? clipLib,
  }) {
    if (clipLib != null) {
      clip_lib = clipLib;
    }
  }

  ffi.DynamicLibrary openLib({
    String? clipLib,
  }) {
    clipLib ??= clip_lib;
    if (Platform.isIOS || Platform.isMacOS) {
      return ffi.DynamicLibrary.process();
    } else {
      return ffi.DynamicLibrary.open(clipLib);
    }
  }

  String native_request(
      {required String modelString,
      required String imageString,
      required String textString,
      String? clipLib}) {
    clipLib ??= clip_lib;
    ffi.Pointer<Utf8> model_string = modelString.toNativeUtf8();
    ffi.Pointer<Utf8> image_string = imageString.toNativeUtf8();
    ffi.Pointer<Utf8> text_string = textString.toNativeUtf8();
    var res = openLib(clipLib: clipLib)
        .lookupFunction<clip_ggml_request_native, clip_ggml_request_native>(
            "test")
        .call(model_string, image_string, text_string);
    //Might need to edit this since this a Utf8 pointer
    return res.toDartString();
  }
}
