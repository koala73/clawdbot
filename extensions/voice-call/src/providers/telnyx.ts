import crypto from "node:crypto";

import type { TelnyxConfig } from "../config.js";
import type { TelnyxMediaStreamHandler } from "../telnyx-media-stream.js";
import type {
  EndReason,
  HangupCallInput,
  InitiateCallInput,
  InitiateCallResult,
  NormalizedEvent,
  PlayTtsInput,
  ProviderWebhookParseResult,
  StartListeningInput,
  StopListeningInput,
  WebhookContext,
  WebhookVerificationResult,
} from "../types.js";
import type { VoiceCallProvider } from "./base.js";
import type { OpenAITTSProvider } from "./tts-openai.js";
import { chunkAudio } from "./tts-openai.js";

/**
 * Telnyx provider options for streaming configuration.
 */
export interface TelnyxProviderOptions {
  /** Public URL for webhook callbacks */
  publicUrl?: string;
  /** Path for media stream WebSocket (e.g., /voice/stream) */
  streamPath?: string;
  /** Request streaming at dial time (more reliable) */
  streamOnDial?: boolean;
  /** Bidirectional streaming mode: 'rtp' or 'raw' */
  bidirectionalMode?: "rtp" | "raw";
  /** Codec for bidirectional audio: PCMU or PCMA */
  bidirectionalCodec?: "PCMU" | "PCMA";
  /** Track to stream (default: both_tracks) */
  streamTrack?: "inbound_track" | "outbound_track" | "both_tracks";
  /** Skip webhook signature verification (development only) */
  skipVerification?: boolean;
}

/**
 * Telnyx Voice API provider implementation.
 *
 * Uses Telnyx Call Control API v2 for managing calls.
 * Supports OpenAI Realtime for high-quality STT/TTS via WebSocket streaming.
 *
 * @see https://developers.telnyx.com/docs/api/v2/call-control
 */
export class TelnyxProvider implements VoiceCallProvider {
  readonly name = "telnyx" as const;

  private readonly apiKey: string;
  private readonly connectionId: string;
  private readonly publicKey: string | undefined;
  private readonly baseUrl = "https://api.telnyx.com/v2";
  private readonly options: TelnyxProviderOptions;

  /** Current public webhook URL */
  private currentPublicUrl: string | null = null;

  /** Optional OpenAI TTS provider for streaming TTS */
  private ttsProvider: OpenAITTSProvider | null = null;

  /** Optional media stream handler for sending audio */
  private mediaStreamHandler: TelnyxMediaStreamHandler | null = null;

  /** Map of providerCallId to streamId for audio routing */
  private callStreamMap = new Map<string, string>();

  constructor(config: TelnyxConfig, options: TelnyxProviderOptions = {}) {
    if (!config.apiKey) {
      throw new Error("Telnyx API key is required");
    }
    if (!config.connectionId) {
      throw new Error("Telnyx connection ID is required");
    }

    this.apiKey = config.apiKey;
    this.connectionId = config.connectionId;
    this.publicKey = config.publicKey;
    this.options = options;

    if (options.publicUrl) {
      this.currentPublicUrl = options.publicUrl;
    }
  }

  /**
   * Set the current public webhook URL (called when tunnel starts).
   */
  setPublicUrl(url: string): void {
    this.currentPublicUrl = url;
  }

  /**
   * Get the current public webhook URL.
   */
  getPublicUrl(): string | null {
    return this.currentPublicUrl;
  }

  /**
   * Set the OpenAI TTS provider for streaming TTS.
   * When set, playTts will use OpenAI audio via media streams.
   */
  setTTSProvider(provider: OpenAITTSProvider): void {
    this.ttsProvider = provider;
  }

  /**
   * Set the media stream handler for sending audio.
   */
  setMediaStreamHandler(handler: TelnyxMediaStreamHandler): void {
    this.mediaStreamHandler = handler;
  }

  /**
   * Register a call's stream ID for audio routing.
   */
  registerCallStream(providerCallId: string, streamId: string): void {
    this.callStreamMap.set(providerCallId, streamId);
  }

  /**
   * Unregister a call's stream ID.
   */
  unregisterCallStream(providerCallId: string): void {
    this.callStreamMap.delete(providerCallId);
  }

  /**
   * Make an authenticated request to the Telnyx API.
   */
  private async apiRequest<T = unknown>(
    endpoint: string,
    body: Record<string, unknown>,
    options?: { allowNotFound?: boolean },
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      if (options?.allowNotFound && response.status === 404) {
        return undefined as T;
      }
      const errorText = await response.text();
      throw new Error(`Telnyx API error: ${response.status} ${errorText}`);
    }

    const text = await response.text();
    return text ? (JSON.parse(text) as T) : (undefined as T);
  }

  /**
   * Verify Telnyx webhook signature using Ed25519.
   */
  verifyWebhook(ctx: WebhookContext): WebhookVerificationResult {
    if (this.options.skipVerification) {
      return { ok: true };
    }

    if (!this.publicKey) {
      // No public key configured, skip verification (not recommended for production)
      return { ok: true };
    }

    const signature = ctx.headers["telnyx-signature-ed25519"];
    const timestamp = ctx.headers["telnyx-timestamp"];

    if (!signature || !timestamp) {
      return { ok: false, reason: "Missing signature or timestamp header" };
    }

    const signatureStr = Array.isArray(signature) ? signature[0] : signature;
    const timestampStr = Array.isArray(timestamp) ? timestamp[0] : timestamp;

    if (!signatureStr || !timestampStr) {
      return { ok: false, reason: "Empty signature or timestamp" };
    }

    try {
      const signedPayload = `${timestampStr}|${ctx.rawBody}`;
      const signatureBuffer = Buffer.from(signatureStr, "base64");
      const publicKeyBuffer = Buffer.from(this.publicKey, "base64");

      const isValid = crypto.verify(
        null, // Ed25519 doesn't use a digest
        Buffer.from(signedPayload),
        {
          key: publicKeyBuffer,
          format: "der",
          type: "spki",
        },
        signatureBuffer,
      );

      if (!isValid) {
        return { ok: false, reason: "Invalid signature" };
      }

      // Check timestamp is within 5 minutes
      const eventTime = parseInt(timestampStr, 10) * 1000;
      const now = Date.now();
      if (Math.abs(now - eventTime) > 5 * 60 * 1000) {
        return { ok: false, reason: "Timestamp too old" };
      }

      return { ok: true };
    } catch (err) {
      return {
        ok: false,
        reason: `Verification error: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  }

  /**
   * Parse Telnyx webhook event into normalized format.
   */
  parseWebhookEvent(ctx: WebhookContext): ProviderWebhookParseResult {
    try {
      const payload = JSON.parse(ctx.rawBody);
      const data = payload.data;

      if (!data || !data.event_type) {
        return { events: [], statusCode: 200 };
      }

      const event = this.normalizeEvent(data);
      return {
        events: event ? [event] : [],
        statusCode: 200,
      };
    } catch {
      return { events: [], statusCode: 400 };
    }
  }

  /**
   * Convert Telnyx event to normalized event format.
   */
  private normalizeEvent(data: TelnyxEvent): NormalizedEvent | null {
    // Decode client_state from Base64 (we encode it in initiateCall)
    let callId = "";
    if (data.payload?.client_state) {
      try {
        callId = Buffer.from(data.payload.client_state, "base64").toString(
          "utf8",
        );
      } catch {
        // Fallback if not valid Base64
        callId = data.payload.client_state;
      }
    }
    if (!callId) {
      callId = data.payload?.call_control_id || "";
    }

    const baseEvent = {
      id: data.id || crypto.randomUUID(),
      callId,
      providerCallId: data.payload?.call_control_id,
      timestamp: Date.now(),
    };

    switch (data.event_type) {
      case "call.initiated":
        return { ...baseEvent, type: "call.initiated" };

      case "call.ringing":
        return { ...baseEvent, type: "call.ringing" };

      case "call.answered":
        return { ...baseEvent, type: "call.answered" };

      case "call.bridged":
        return { ...baseEvent, type: "call.active" };

      case "call.speak.started":
        return {
          ...baseEvent,
          type: "call.speaking",
          text: data.payload?.text || "",
        };

      case "call.transcription":
        return {
          ...baseEvent,
          type: "call.speech",
          transcript: data.payload?.transcription || "",
          isFinal: data.payload?.is_final ?? true,
          confidence: data.payload?.confidence,
        };

      case "call.hangup":
        return {
          ...baseEvent,
          type: "call.ended",
          reason: this.mapHangupCause(data.payload?.hangup_cause),
        };

      case "call.dtmf.received":
        return {
          ...baseEvent,
          type: "call.dtmf",
          digits: data.payload?.digit || "",
        };

      // Streaming events (for media stream lifecycle)
      case "streaming.started":
        return { ...baseEvent, type: "call.streaming.started" };

      case "streaming.stopped":
        return {
          ...baseEvent,
          type: "call.streaming.stopped",
          reason: data.payload?.reason || "stopped",
        };

      case "streaming.failed":
        return {
          ...baseEvent,
          type: "call.streaming.failed",
          reason: data.payload?.reason || "failed",
        };

      default:
        return null;
    }
  }

  /**
   * Map Telnyx hangup cause to normalized end reason.
   * @see https://developers.telnyx.com/docs/api/v2/call-control/Call-Commands#hangup-causes
   */
  private mapHangupCause(cause?: string): EndReason {
    switch (cause) {
      case "normal_clearing":
      case "normal_unspecified":
        return "completed";
      case "originator_cancel":
        return "hangup-bot";
      case "call_rejected":
      case "user_busy":
        return "busy";
      case "no_answer":
      case "no_user_response":
        return "no-answer";
      case "destination_out_of_order":
      case "network_out_of_order":
      case "service_unavailable":
      case "recovery_on_timer_expire":
        return "failed";
      case "machine_detected":
      case "fax_detected":
        return "voicemail";
      case "user_hangup":
      case "subscriber_absent":
        return "hangup-user";
      default:
        // Unknown cause - log it for debugging and return completed
        if (cause) {
          console.warn(`[telnyx] Unknown hangup cause: ${cause}`);
        }
        return "completed";
    }
  }

  /**
   * Get the WebSocket URL for media streaming.
   */
  private getStreamUrl(): string | null {
    if (!this.currentPublicUrl || !this.options.streamPath) {
      return null;
    }

    const url = new URL(this.currentPublicUrl);
    const origin = url.origin;

    // Convert https:// to wss:// for WebSocket
    const wsOrigin = origin
      .replace(/^https:\/\//, "wss://")
      .replace(/^http:\/\//, "ws://");

    const path = this.options.streamPath.startsWith("/")
      ? this.options.streamPath
      : `/${this.options.streamPath}`;

    return `${wsOrigin}${path}`;
  }

  /**
   * Initiate an outbound call via Telnyx API.
   * Optionally includes stream_url for bidirectional audio at dial time.
   */
  async initiateCall(input: InitiateCallInput): Promise<InitiateCallResult> {
    const streamUrl = this.options.streamOnDial ? this.getStreamUrl() : null;

    const requestBody: Record<string, unknown> = {
      connection_id: this.connectionId,
      to: input.to,
      from: input.from,
      webhook_url: input.webhookUrl,
      webhook_url_method: "POST",
      client_state: Buffer.from(input.callId).toString("base64"),
      timeout_secs: 30,
    };

    // Include stream configuration at dial time if enabled
    if (streamUrl) {
      requestBody.stream_url = streamUrl;
      requestBody.stream_track = this.options.streamTrack || "both_tracks";

      if (this.options.bidirectionalMode) {
        requestBody.stream_bidirectional_mode = this.options.bidirectionalMode;
      }
      if (this.options.bidirectionalCodec) {
        requestBody.stream_bidirectional_codec =
          this.options.bidirectionalCodec;
      }

      console.log(`[telnyx] Initiating call with stream_url: ${streamUrl}`);
    }

    const result = await this.apiRequest<TelnyxCallResponse>(
      "/calls",
      requestBody,
    );

    return {
      providerCallId: result.data.call_control_id,
      status: "initiated",
    };
  }

  /**
   * Start media streaming on an active call.
   * Call this after call.answered if not using streamOnDial.
   */
  async startStreaming(providerCallId: string): Promise<void> {
    const streamUrl = this.getStreamUrl();
    if (!streamUrl) {
      throw new Error("Stream URL not configured");
    }

    const requestBody: Record<string, unknown> = {
      stream_url: streamUrl,
      stream_track: this.options.streamTrack || "both_tracks",
    };

    if (this.options.bidirectionalMode) {
      requestBody.stream_bidirectional_mode = this.options.bidirectionalMode;
    }
    if (this.options.bidirectionalCodec) {
      requestBody.stream_bidirectional_codec = this.options.bidirectionalCodec;
    }

    console.log(`[telnyx] Starting streaming for ${providerCallId}`);

    await this.apiRequest(
      `/calls/${providerCallId}/actions/streaming_start`,
      requestBody,
    );
  }

  /**
   * Hang up a call via Telnyx API.
   */
  async hangupCall(input: HangupCallInput): Promise<void> {
    this.callStreamMap.delete(input.providerCallId);

    await this.apiRequest(
      `/calls/${input.providerCallId}/actions/hangup`,
      { command_id: crypto.randomUUID() },
      { allowNotFound: true },
    );
  }

  /**
   * Play TTS audio.
   *
   * Two modes:
   * 1. OpenAI TTS + Media Streams: If TTS provider and media stream are available,
   *    generates audio via OpenAI and streams it through WebSocket (preferred).
   * 2. Native Telnyx TTS: Falls back to Telnyx's built-in speak action.
   */
  async playTts(input: PlayTtsInput): Promise<void> {
    // Try OpenAI TTS via media stream first (if configured)
    const streamId = this.callStreamMap.get(input.providerCallId);
    if (this.ttsProvider && this.mediaStreamHandler && streamId) {
      try {
        await this.playTtsViaStream(input.text, streamId);
        return;
      } catch (err) {
        console.warn(
          `[telnyx] OpenAI TTS failed, falling back to native TTS:`,
          err instanceof Error ? err.message : err,
        );
        // Fall through to native Telnyx TTS
      }
    }

    // Fall back to native Telnyx speak action
    await this.apiRequest(`/calls/${input.providerCallId}/actions/speak`, {
      command_id: crypto.randomUUID(),
      payload: input.text,
      voice: input.voice || "female",
      language: input.locale || "en-US",
    });
  }

  /**
   * Play TTS via OpenAI and Telnyx Media Streams.
   * Generates audio with OpenAI TTS, converts to mu-law, and streams via WebSocket.
   */
  private async playTtsViaStream(text: string, streamId: string): Promise<void> {
    if (!this.ttsProvider || !this.mediaStreamHandler) {
      throw new Error("TTS provider and media stream handler required");
    }

    // Generate audio with OpenAI TTS (returns mu-law at 8kHz)
    const muLawAudio = await this.ttsProvider.synthesizeForTwilio(text);

    // Stream audio in 20ms chunks (160 bytes at 8kHz mu-law)
    const CHUNK_SIZE = 160;
    const CHUNK_DELAY_MS = 20;

    for (const chunk of chunkAudio(muLawAudio, CHUNK_SIZE)) {
      this.mediaStreamHandler.sendAudio(streamId, chunk);

      // Pace the audio to match real-time playback
      await new Promise((resolve) => setTimeout(resolve, CHUNK_DELAY_MS));
    }

    // Send a mark to track when audio finishes
    this.mediaStreamHandler.sendMark(streamId, `tts-${Date.now()}`);
  }

  /**
   * Start transcription (STT).
   * When using OpenAI Realtime via media streams, this is handled by the stream handler.
   * Falls back to native Telnyx transcription.
   */
  async startListening(input: StartListeningInput): Promise<void> {
    // If media streaming is active, STT is handled by the stream handler
    const streamId = this.callStreamMap.get(input.providerCallId);
    if (this.mediaStreamHandler && streamId) {
      // STT is already running via OpenAI Realtime
      console.log(`[telnyx] STT handled by media stream for ${streamId}`);
      return;
    }

    // Fall back to native Telnyx transcription
    await this.apiRequest(
      `/calls/${input.providerCallId}/actions/transcription_start`,
      {
        command_id: crypto.randomUUID(),
        language: input.language || "en",
      },
    );
  }

  /**
   * Stop transcription via Telnyx.
   */
  async stopListening(input: StopListeningInput): Promise<void> {
    await this.apiRequest(
      `/calls/${input.providerCallId}/actions/transcription_stop`,
      { command_id: crypto.randomUUID() },
      { allowNotFound: true },
    );
  }
}

// -----------------------------------------------------------------------------
// Telnyx-specific types
// -----------------------------------------------------------------------------

interface TelnyxEvent {
  id?: string;
  event_type: string;
  payload?: {
    call_control_id?: string;
    client_state?: string;
    text?: string;
    transcription?: string;
    is_final?: boolean;
    confidence?: number;
    hangup_cause?: string;
    digit?: string;
    reason?: string;
    [key: string]: unknown;
  };
}

interface TelnyxCallResponse {
  data: {
    call_control_id: string;
    call_leg_id: string;
    call_session_id: string;
    is_alive: boolean;
    record_type: string;
  };
}
