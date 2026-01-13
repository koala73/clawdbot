/**
 * Audio Codec Utilities
 *
 * Conversion functions for telephony audio formats:
 * - PCM ↔ μ-law (G.711 PCMU)
 * - A-law → μ-law (G.711 PCMA → PCMU)
 * - RTP header wrapping for Telnyx bidirectional mode
 */

/**
 * Convert a single PCM sample to μ-law.
 */
export function pcmToMuLawSample(pcm: number): number {
  const BIAS = 0x84;
  const CLIP = 32635;
  let sign = (pcm >> 8) & 0x80;
  if (sign) pcm = -pcm;
  if (pcm > CLIP) pcm = CLIP;
  pcm += BIAS;
  let exponent = 7;
  for (let expMask = 0x4000; (pcm & expMask) === 0 && exponent > 0; exponent--) {
    expMask >>= 1;
  }
  const mantissa = (pcm >> (exponent + 3)) & 0x0f;
  return (~(sign | (exponent << 4) | mantissa)) & 0xff;
}

/**
 * Convert PCM buffer (16-bit signed LE) to μ-law.
 */
export function pcmToMuLaw(pcmData: Buffer): Buffer {
  const muLawData = Buffer.alloc(Math.floor(pcmData.length / 2));
  for (let i = 0; i < muLawData.length; i++) {
    const pcm = pcmData.readInt16LE(i * 2);
    muLawData[i] = pcmToMuLawSample(pcm);
  }
  return muLawData;
}

/**
 * Convert a single A-law sample to PCM.
 */
export function alawToPcmSample(aLawByte: number): number {
  let a = (aLawByte ^ 0x55) & 0xff;
  const sign = a & 0x80;
  const exponent = (a & 0x70) >> 4;
  const mantissa = a & 0x0f;

  let sample = (mantissa << 4) + 8;
  if (exponent !== 0) sample += 0x100;
  if (exponent > 1) sample <<= exponent - 1;

  return sign ? sample : -sample;
}

/**
 * Convert A-law buffer to μ-law buffer.
 * Telnyx often negotiates PCMA (A-law), but OpenAI Realtime expects μ-law.
 */
export function alawToMulaw(aLaw: Buffer): Buffer {
  const out = Buffer.alloc(aLaw.length);
  for (let i = 0; i < aLaw.length; i++) {
    const pcm = alawToPcmSample(aLaw[i]);
    out[i] = pcmToMuLawSample(pcm);
  }
  return out;
}

/**
 * State for RTP header generation.
 */
export interface RtpState {
  rtpSeq: number;
  rtpTimestamp: number;
  rtpSsrc: number;
  codec: "PCMU" | "PCMA";
}

/**
 * Create initial RTP state.
 */
export function createRtpState(codec: "PCMU" | "PCMA" = "PCMU"): RtpState {
  return {
    rtpSeq: 0,
    rtpTimestamp: 0,
    rtpSsrc: Math.floor(Math.random() * 0xffffffff),
    codec,
  };
}

/**
 * Wrap audio payload with RTP header for Telnyx bidirectional mode.
 *
 * RTP Header (12 bytes):
 * - Byte 0: V=2, P=0, X=0, CC=0 (0x80)
 * - Byte 1: M=0, PT (PCMU=0, PCMA=8)
 * - Bytes 2-3: Sequence number (big-endian)
 * - Bytes 4-7: Timestamp (big-endian)
 * - Bytes 8-11: SSRC (big-endian)
 */
export function wrapRtpHeader(state: RtpState, payload: Buffer): Buffer {
  const header = Buffer.alloc(12);

  // V=2, P=0, X=0, CC=0
  header[0] = 0x80;

  // M=0, PT (PCMU=0, PCMA=8)
  const pt = state.codec === "PCMA" ? 8 : 0;
  header[1] = pt & 0x7f;

  // Sequence number
  header.writeUInt16BE(state.rtpSeq & 0xffff, 2);

  // Timestamp
  header.writeUInt32BE(state.rtpTimestamp >>> 0, 4);

  // SSRC
  header.writeUInt32BE(state.rtpSsrc >>> 0, 8);

  // Advance state for next packet
  state.rtpSeq = (state.rtpSeq + 1) & 0xffff;
  // For G.711, 1 byte = 1 sample at 8kHz
  state.rtpTimestamp = (state.rtpTimestamp + payload.length) >>> 0;

  return Buffer.concat([header, payload]);
}

/**
 * Extract audio payload from inbound message.
 * Handles both JSON-wrapped (Twilio/Telnyx) and raw binary formats.
 */
export function extractInboundAudio(
  msgBuffer: Buffer,
  expectedStreamId: string | null,
): Buffer | null {
  if (msgBuffer.length === 0) return null;

  // Raw binary audio (not JSON)
  if (msgBuffer[0] !== 0x7b) {
    // Heuristic: looks like audio if it's a multiple of 160 bytes (20ms at 8kHz)
    const looksLikeAudio =
      msgBuffer.length <= 160 * 50 && msgBuffer.length % 160 === 0;
    const hasStream = !!expectedStreamId;
    return looksLikeAudio && hasStream ? msgBuffer : null;
  }

  // JSON-wrapped audio
  try {
    const msg = JSON.parse(msgBuffer.toString()) as {
      event?: string;
      type?: string;
      stream_id?: string;
      streamSid?: string;
      streamId?: string;
      media?: {
        payload?: string;
        track?: string;
      };
      payload?: string;
      track?: string;
    };

    const eventType = msg.event || msg.type;
    if (eventType === "media") {
      const payload = msg.media?.payload || msg.payload;
      if (!payload) return null;

      // Validate stream ID matches
      const sid = msg.stream_id || msg.streamSid || msg.streamId;
      if (expectedStreamId && sid && sid !== expectedStreamId) return null;

      // Filter by track (only inbound audio)
      const track = msg.media?.track || msg.track;
      if (
        track &&
        !["inbound", "inbound_track", "inbound_audio"].includes(track)
      ) {
        return null;
      }

      return Buffer.from(payload, "base64");
    }
  } catch {
    // Not valid JSON, ignore
  }

  return null;
}
