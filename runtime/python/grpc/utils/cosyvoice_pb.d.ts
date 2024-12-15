// package: cosyvoice
// file: cosyvoice.proto

import * as jspb from "google-protobuf";

export class Request extends jspb.Message {
  hasSftRequest(): boolean;
  clearSftRequest(): void;
  getSftRequest(): sftRequest | undefined;
  setSftRequest(value?: sftRequest): void;

  hasZeroShotRequest(): boolean;
  clearZeroShotRequest(): void;
  getZeroShotRequest(): zeroshotRequest | undefined;
  setZeroShotRequest(value?: zeroshotRequest): void;

  hasCrossLingualRequest(): boolean;
  clearCrossLingualRequest(): void;
  getCrossLingualRequest(): crosslingualRequest | undefined;
  setCrossLingualRequest(value?: crosslingualRequest): void;

  hasInstructRequest(): boolean;
  clearInstructRequest(): void;
  getInstructRequest(): instructRequest | undefined;
  setInstructRequest(value?: instructRequest): void;

  getRequestpayloadCase(): Request.RequestpayloadCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Request.AsObject;
  static toObject(includeInstance: boolean, msg: Request): Request.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Request, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Request;
  static deserializeBinaryFromReader(message: Request, reader: jspb.BinaryReader): Request;
}

export namespace Request {
  export type AsObject = {
    sftRequest?: sftRequest.AsObject,
    zeroShotRequest?: zeroshotRequest.AsObject,
    crossLingualRequest?: crosslingualRequest.AsObject,
    instructRequest?: instructRequest.AsObject,
  }

  export enum RequestpayloadCase {
    REQUESTPAYLOAD_NOT_SET = 0,
    SFT_REQUEST = 1,
    ZERO_SHOT_REQUEST = 2,
    CROSS_LINGUAL_REQUEST = 3,
    INSTRUCT_REQUEST = 4,
  }
}

export class sftRequest extends jspb.Message {
  getSpkId(): string;
  setSpkId(value: string): void;

  getTtsText(): string;
  setTtsText(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): sftRequest.AsObject;
  static toObject(includeInstance: boolean, msg: sftRequest): sftRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: sftRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): sftRequest;
  static deserializeBinaryFromReader(message: sftRequest, reader: jspb.BinaryReader): sftRequest;
}

export namespace sftRequest {
  export type AsObject = {
    spkId: string,
    ttsText: string,
  }
}

export class zeroshotRequest extends jspb.Message {
  getTtsText(): string;
  setTtsText(value: string): void;

  getPromptText(): string;
  setPromptText(value: string): void;

  getPromptAudio(): Uint8Array | string;
  getPromptAudio_asU8(): Uint8Array;
  getPromptAudio_asB64(): string;
  setPromptAudio(value: Uint8Array | string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): zeroshotRequest.AsObject;
  static toObject(includeInstance: boolean, msg: zeroshotRequest): zeroshotRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: zeroshotRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): zeroshotRequest;
  static deserializeBinaryFromReader(message: zeroshotRequest, reader: jspb.BinaryReader): zeroshotRequest;
}

export namespace zeroshotRequest {
  export type AsObject = {
    ttsText: string,
    promptText: string,
    promptAudio: Uint8Array | string,
  }
}

export class crosslingualRequest extends jspb.Message {
  getTtsText(): string;
  setTtsText(value: string): void;

  getPromptAudio(): Uint8Array | string;
  getPromptAudio_asU8(): Uint8Array;
  getPromptAudio_asB64(): string;
  setPromptAudio(value: Uint8Array | string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): crosslingualRequest.AsObject;
  static toObject(includeInstance: boolean, msg: crosslingualRequest): crosslingualRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: crosslingualRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): crosslingualRequest;
  static deserializeBinaryFromReader(message: crosslingualRequest, reader: jspb.BinaryReader): crosslingualRequest;
}

export namespace crosslingualRequest {
  export type AsObject = {
    ttsText: string,
    promptAudio: Uint8Array | string,
  }
}

export class instructRequest extends jspb.Message {
  getTtsText(): string;
  setTtsText(value: string): void;

  getSpkId(): string;
  setSpkId(value: string): void;

  getInstructText(): string;
  setInstructText(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): instructRequest.AsObject;
  static toObject(includeInstance: boolean, msg: instructRequest): instructRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: instructRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): instructRequest;
  static deserializeBinaryFromReader(message: instructRequest, reader: jspb.BinaryReader): instructRequest;
}

export namespace instructRequest {
  export type AsObject = {
    ttsText: string,
    spkId: string,
    instructText: string,
  }
}

export class Response extends jspb.Message {
  getTtsAudio(): Uint8Array | string;
  getTtsAudio_asU8(): Uint8Array;
  getTtsAudio_asB64(): string;
  setTtsAudio(value: Uint8Array | string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Response.AsObject;
  static toObject(includeInstance: boolean, msg: Response): Response.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Response, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Response;
  static deserializeBinaryFromReader(message: Response, reader: jspb.BinaryReader): Response;
}

export namespace Response {
  export type AsObject = {
    ttsAudio: Uint8Array | string,
  }
}

