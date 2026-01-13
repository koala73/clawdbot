/**
 * Telnyx Media Stream Handler
 *
 * Handles bidirectional audio streaming between Telnyx and OpenAI Realtime.
 *
 * Key differences from Twilio:
 * - Uses `stream_id` instead of `streamSid`
 * - Message format: { event: 'media', stream_id: '...', media: { payload: '...' } }
 * - Codec negotiation (PCMA vs PCMU) - need A-law → μ-law conversion
 * - For bidirectional RTP mode: wrap payload with RTP header
 */

import type { IncomingMessage } from "node:http";
import type { Duplex } from "node:stream";

import { WebSocket, WebSocketServer } from "ws";

import {
  alawToMulaw,
  createRtpState,
  extractInboundAudio,
  type RtpState,
  wrapRtpHeader,
} from "./audio-utils.js";
import type {
  OpenAIRealtimeSTTProvider,
  RealtimeSTTSession,
} from "./providers/stt-openai-realtime.js";

/**
 * Configuration for the Telnyx media stream handler.
 */
export interface TelnyxMediaStreamConfig {
  /** STT provider for transcription */
  sttProvider: OpenAIRealtimeSTTProvider;
  /** Callback when transcript is received */
  onTranscript?: (callId: string, transcript: string) => void;
  /** Callback for partial transcripts (streaming UI) */
  onPartialTranscript?: (callId: string, partial: string) => void;
  /** Callback when stream connects */
  onConnect?: (callId: string, streamId: string) => void;
  /** Callback when stream disconnects */
  onDisconnect?: (callId: string) => void;
  /** Enable RTP header wrapping for bidirectional mode */
  bidirectionalMode?: "rtp" | "raw";
  /** Codec for outbound audio (PCMU or PCMA) */
  bidirectionalCodec?: "PCMU" | "PCMA";
}

/**
 * Active Telnyx media stream session.
 */
interface TelnyxStreamSession {
  callId: string;
  streamId: string;
  ws: WebSocket;
  sttSession: RealtimeSTTSession;
  /** Negotiated inbound codec from Telnyx */
  inboundCodec: "PCMU" | "PCMA" | null;
  /** Outbound codec for TTS */
  outboundCodec: "PCMU" | "PCMA";
  /** RTP state for header generation */
  rtpState: RtpState | null;
  /** Whether we've logged the start event (avoid spam) */
  startLogged: boolean;
  /** Whether we've logged media (avoid spam) */
  mediaLogged: boolean;
}

/**
 * Telnyx WebSocket message types.
 */
interface TelnyxMediaMessage {
  event: "connected" | "start" | "media" | "stop" | "error";
  stream_id?: string;
  streamId?: string;
  start?: {
    stream_id?: string;
    streamId?: string;
    call_control_id?: string;
    media_format?: {
      encoding?: string;
      sample_rate?: number;
      channels?: number;
    };
  };
  media?: {
    track?: string;
    payload?: string;
  };
  payload?: string;
  reason?: string;
  error?: string;
}

/**
 * Manages WebSocket connections for Telnyx media streams.
 */
export class TelnyxMediaStreamHandler {
  private wss: WebSocketServer | null = null;
  private sessions = new Map<string, TelnyxStreamSession>();
  private config: TelnyxMediaStreamConfig;

  constructor(config: TelnyxMediaStreamConfig) {
    this.config = config;
  }

  /**
   * Handle WebSocket upgrade for media stream connections.
   */
  handleUpgrade(request: IncomingMessage, socket: Duplex, head: Buffer): void {
    if (!this.wss) {
      this.wss = new WebSocketServer({ noServer: true });
      this.wss.on("connection", (ws, req) => this.handleConnection(ws, req));
    }

    this.wss.handleUpgrade(request, socket, head, (ws) => {
      this.wss?.emit("connection", ws, request);
    });
  }

  /**
   * Handle new WebSocket connection from Telnyx.
   */
  private async handleConnection(
    ws: WebSocket,
    _request: IncomingMessage,
  ): Promise<void> {
    let session: TelnyxStreamSession | null = null;

    ws.on("message", async (data: Buffer) => {
      try {
        // Try to parse as JSON
        const message = JSON.parse(data.toString()) as TelnyxMediaMessage;

        switch (message.event) {
          case "connected":
            console.log("[TelnyxMediaStream] WebSocket connected");
            break;

          case "start":
            session = await this.handleStart(ws, message);
            break;

          case "media":
            if (session) {
              this.handleMedia(session, message, data);
            }
            break;

          case "stop":
            if (session) {
              this.handleStop(session);
              session = null;
            }
            break;

          case "error":
            console.error(
              `[TelnyxMediaStream] Error event: ${message.error || message.reason}`,
            );
            break;
        }
      } catch {
        // Not JSON - might be raw audio in some modes
        if (session) {
          const audioData = extractInboundAudio(data, session.streamId);
          if (audioData) {
            this.processAudio(session, audioData);
          }
        }
      }
    });

    ws.on("close", () => {
      if (session) {
        this.handleStop(session);
      }
    });

    ws.on("error", (error) => {
      console.error("[TelnyxMediaStream] WebSocket error:", error);
    });
  }

  /**
   * Handle stream start event.
   */
  private async handleStart(
    ws: WebSocket,
    message: TelnyxMediaMessage,
  ): Promise<TelnyxStreamSession> {
    const streamId =
      message.stream_id ||
      message.streamId ||
      message.start?.stream_id ||
      message.start?.streamId ||
      "";
    const callControlId = message.start?.call_control_id || "";

    // Detect codec from start event
    const encoding = message.start?.media_format?.encoding?.toUpperCase();
    const inboundCodec: "PCMU" | "PCMA" | null =
      encoding === "PCMA" ? "PCMA" : encoding === "PCMU" ? "PCMU" : null;

    const outboundCodec = this.config.bidirectionalCodec || "PCMU";

    console.log(
      `[TelnyxMediaStream] Stream started: ${streamId} (call: ${callControlId}, codec: ${encoding || "unknown"})`,
    );

    // Create STT session
    const sttSession = this.config.sttProvider.createSession();

    // Set up transcript callbacks
    sttSession.onPartial((partial) => {
      this.config.onPartialTranscript?.(callControlId || streamId, partial);
    });

    sttSession.onTranscript((transcript) => {
      this.config.onTranscript?.(callControlId || streamId, transcript);
    });

    // Create RTP state if bidirectional RTP mode
    const rtpState =
      this.config.bidirectionalMode === "rtp"
        ? createRtpState(outboundCodec)
        : null;

    const session: TelnyxStreamSession = {
      callId: callControlId || streamId,
      streamId,
      ws,
      sttSession,
      inboundCodec,
      outboundCodec,
      rtpState,
      startLogged: false,
      mediaLogged: false,
    };

    this.sessions.set(streamId, session);

    // Notify connection BEFORE STT connect
    this.config.onConnect?.(callControlId || streamId, streamId);

    // Connect to OpenAI STT (non-blocking)
    sttSession.connect().catch((err) => {
      console.warn(
        `[TelnyxMediaStream] STT connection failed (TTS still works):`,
        err instanceof Error ? err.message : err,
      );
    });

    return session;
  }

  /**
   * Handle media event with audio payload.
   */
  private handleMedia(
    session: TelnyxStreamSession,
    message: TelnyxMediaMessage,
    rawData: Buffer,
  ): void {
    // Log first media event for debugging
    if (!session.mediaLogged) {
      session.mediaLogged = true;
      const track = message.media?.track;
      console.log(
        `[TelnyxMediaStream] First media: stream_id=${session.streamId} track=${track || "—"}`,
      );
    }

    // Extract audio from message
    const audioData = extractInboundAudio(rawData, session.streamId);
    if (audioData) {
      this.processAudio(session, audioData);
    }
  }

  /**
   * Process inbound audio - convert codec if needed and forward to STT.
   */
  private processAudio(session: TelnyxStreamSession, audioData: Buffer): void {
    let processedAudio = audioData;

    // Convert A-law to μ-law if Telnyx negotiated PCMA
    // OpenAI Realtime expects g711_ulaw
    if (session.inboundCodec === "PCMA") {
      processedAudio = alawToMulaw(audioData);
    }

    // Forward to STT
    session.sttSession.sendAudio(processedAudio);
  }

  /**
   * Handle stream stop event.
   */
  private handleStop(session: TelnyxStreamSession): void {
    console.log(`[TelnyxMediaStream] Stream stopped: ${session.streamId}`);

    session.sttSession.close();
    this.sessions.delete(session.streamId);
    this.config.onDisconnect?.(session.callId);
  }

  /**
   * Get session by stream ID.
   */
  getSession(streamId: string): TelnyxStreamSession | undefined {
    return this.sessions.get(streamId);
  }

  /**
   * Get session by call ID.
   */
  getSessionByCallId(callId: string): TelnyxStreamSession | undefined {
    return [...this.sessions.values()].find(
      (session) => session.callId === callId,
    );
  }

  /**
   * Send audio to a specific stream (for TTS playback).
   * Audio should be μ-law encoded at 8kHz mono.
   */
  sendAudio(streamId: string, muLawAudio: Buffer): void {
    const session = this.sessions.get(streamId);
    if (!session || session.ws.readyState !== WebSocket.OPEN) return;

    let payload = muLawAudio;

    // Wrap with RTP header if in bidirectional RTP mode
    if (session.rtpState) {
      payload = wrapRtpHeader(session.rtpState, muLawAudio);
    }

    // Send as Telnyx media message
    const message: Record<string, unknown> = {
      event: "media",
      stream_id: streamId,
      media: {
        payload: payload.toString("base64"),
      },
    };

    try {
      session.ws.send(JSON.stringify(message));
    } catch (err) {
      console.warn(
        `[TelnyxMediaStream] Failed to send audio:`,
        err instanceof Error ? err.message : err,
      );
    }
  }

  /**
   * Send a mark event (for tracking playback position).
   */
  sendMark(streamId: string, name: string): void {
    const session = this.sessions.get(streamId);
    if (!session || session.ws.readyState !== WebSocket.OPEN) return;

    try {
      session.ws.send(
        JSON.stringify({
          event: "mark",
          stream_id: streamId,
          mark: { name },
        }),
      );
    } catch {
      // Ignore errors
    }
  }

  /**
   * Clear audio buffer (interrupt playback).
   */
  clearAudio(streamId: string): void {
    const session = this.sessions.get(streamId);
    if (!session || session.ws.readyState !== WebSocket.OPEN) return;

    try {
      session.ws.send(
        JSON.stringify({
          event: "clear",
          stream_id: streamId,
        }),
      );
    } catch {
      // Ignore errors
    }
  }

  /**
   * Close all sessions.
   */
  closeAll(): void {
    for (const session of this.sessions.values()) {
      session.sttSession.close();
      session.ws.close();
    }
    this.sessions.clear();
  }
}
