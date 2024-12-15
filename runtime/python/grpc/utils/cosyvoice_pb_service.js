// package: cosyvoice
// file: cosyvoice.proto

var cosyvoice_pb = require("./cosyvoice_pb");
var grpc = require("@improbable-eng/grpc-web").grpc;

var CosyVoice = (function () {
  function CosyVoice() {}
  CosyVoice.serviceName = "cosyvoice.CosyVoice";
  return CosyVoice;
}());

CosyVoice.Inference = {
  methodName: "Inference",
  service: CosyVoice,
  requestStream: false,
  responseStream: true,
  requestType: cosyvoice_pb.Request,
  responseType: cosyvoice_pb.Response
};

exports.CosyVoice = CosyVoice;

function CosyVoiceClient(serviceHost, options) {
  this.serviceHost = serviceHost;
  this.options = options || {};
}

CosyVoiceClient.prototype.inference = function inference(requestMessage, metadata) {
  var listeners = {
    data: [],
    end: [],
    status: []
  };
  var client = grpc.invoke(CosyVoice.Inference, {
    request: requestMessage,
    host: this.serviceHost,
    metadata: metadata,
    transport: this.options.transport,
    debug: this.options.debug,
    onMessage: function (responseMessage) {
      listeners.data.forEach(function (handler) {
        handler(responseMessage);
      });
    },
    onEnd: function (status, statusMessage, trailers) {
      listeners.status.forEach(function (handler) {
        handler({ code: status, details: statusMessage, metadata: trailers });
      });
      listeners.end.forEach(function (handler) {
        handler({ code: status, details: statusMessage, metadata: trailers });
      });
      listeners = null;
    }
  });
  return {
    on: function (type, handler) {
      listeners[type].push(handler);
      return this;
    },
    cancel: function () {
      listeners = null;
      client.close();
    }
  };
};

exports.CosyVoiceClient = CosyVoiceClient;

