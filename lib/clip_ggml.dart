import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:ffi/ffi.dart';

typedef clip_ggml_request_native = ffi.Pointer<Utf8> Function(ffi.Pointer<Utf8> input);

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

  String native_request({
    required String inputString,
    String? clipLib
  }) {
    clipLib ??= clip_lib
    ffi.Pointer<Utf8> input_string = inputString.toNativeUtf8();
    var res = openLib(clipLib: clipLib).lookup(clip_ggml_request_native, clip_ggml_request_native)("test").call(input_string);
    //Might need to edit this since this a Utf8 pointer
    return res;
  }
}
