/**
 * WhatsApp client wrapper using Baileys.
 * Based on OpenClaw's working implementation.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
import makeWASocket, {
  DisconnectReason,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  makeCacheableSignalKeyStore,
} from '@whiskeysockets/baileys';

import { Boom } from '@hapi/boom';
import qrcode from 'qrcode-terminal';
import pino from 'pino';
import { HttpsProxyAgent } from 'https-proxy-agent';

const VERSION = '0.1.0';

export interface InboundMessage {
  id: string;
  sender: string;
  pn: string;
  content: string;
  timestamp: number;
  isGroup: boolean;
  /** Text of the quoted/replied-to message, if any */
  quotedText?: string;
}

export interface WhatsAppClientOptions {
  authDir: string;
  onMessage: (msg: InboundMessage) => void;
  onQR: (qr: string) => void;
  onStatus: (status: string) => void;
}

export class WhatsAppClient {
  private sock: any = null;
  private options: WhatsAppClientOptions;
  private reconnecting = false;
  private _sentIds: Set<string> = new Set();
  /** Cache recent inbound messages so we can quote them in replies */
  private _recentMessages: Map<string, any> = new Map();

  constructor(options: WhatsAppClientOptions) {
    this.options = options;
  }

  async connect(): Promise<void> {
    const logger = pino({ level: 'silent' });
    const { state, saveCreds } = await useMultiFileAuthState(this.options.authDir);
    const { version } = await fetchLatestBaileysVersion();

    console.log(`Using Baileys version: ${version.join('.')}`);

    // Proxy support: use HTTPS_PROXY, HTTP_PROXY, or default to Clash mixed port
    const proxyUrl = process.env.HTTPS_PROXY || process.env.HTTP_PROXY || process.env.ALL_PROXY;
    let agent: any = undefined;
    if (proxyUrl) {
      agent = new HttpsProxyAgent(proxyUrl);
      console.log(`🔀 Using proxy: ${proxyUrl}`);
    } else {
      // Auto-detect: try Clash default mixed port
      const defaultProxy = 'http://127.0.0.1:7890';
      try {
        const net = await import('net');
        await new Promise<void>((resolve, reject) => {
          const s = net.createConnection({ host: '127.0.0.1', port: 7890 }, () => { s.end(); resolve(); });
          s.on('error', reject);
          s.setTimeout(1000, () => { s.destroy(); reject(new Error('timeout')); });
        });
        agent = new HttpsProxyAgent(defaultProxy);
        console.log(`🔀 Auto-detected proxy: ${defaultProxy}`);
      } catch {
        console.log('ℹ️  No proxy detected, connecting directly');
      }
    }

    // Create socket following OpenClaw's pattern
    this.sock = makeWASocket({
      auth: {
        creds: state.creds,
        keys: makeCacheableSignalKeyStore(state.keys, logger),
      },
      version,
      logger,
      printQRInTerminal: false,
      browser: ['nanobot', 'cli', VERSION],
      syncFullHistory: false,
      markOnlineOnConnect: false,
      agent,
    });

    // Handle WebSocket errors
    if (this.sock.ws && typeof this.sock.ws.on === 'function') {
      this.sock.ws.on('error', (err: Error) => {
        console.error('WebSocket error:', err.message);
      });
    }

    // Handle connection updates
    this.sock.ev.on('connection.update', async (update: any) => {
      const { connection, lastDisconnect, qr } = update;

      if (qr) {
        // Display QR code in terminal
        console.log('\n📱 Scan this QR code with WhatsApp (Linked Devices):\n');
        qrcode.generate(qr, { small: true });
        this.options.onQR(qr);
      }

      if (connection === 'close') {
        const statusCode = (lastDisconnect?.error as Boom)?.output?.statusCode;
        const shouldReconnect = statusCode !== DisconnectReason.loggedOut;

        console.log(`Connection closed. Status: ${statusCode}, Will reconnect: ${shouldReconnect}`);
        this.options.onStatus('disconnected');

        if (shouldReconnect && !this.reconnecting) {
          this.reconnecting = true;
          console.log('Reconnecting in 5 seconds...');
          setTimeout(() => {
            this.reconnecting = false;
            this.connect();
          }, 5000);
        }
      } else if (connection === 'open') {
        console.log('✅ Connected to WhatsApp');
        this.options.onStatus('connected');
      }
    });

    // Save credentials on update
    this.sock.ev.on('creds.update', saveCreds);

    // Handle incoming messages
    this.sock.ev.on('messages.upsert', async ({ messages, type }: { messages: any[]; type: string }) => {
      if (type !== 'notify') return;

      for (const msg of messages) {
        // Skip status updates
        if (msg.key.remoteJid === 'status@broadcast') continue;

        // Track messages sent by the bridge to avoid echo loops
        if (msg.key.fromMe && this._sentIds.has(msg.key.id || '')) {
          this._sentIds.delete(msg.key.id || '');
          continue;
        }

        const extracted = this.extractMessageContent(msg);
        if (!extracted) continue;

        // Store the raw message for potential quoting later
        this._recentMessages.set(msg.key.id || '', msg);
        // Evict old entries (keep last 200)
        if (this._recentMessages.size > 200) {
          const oldest = this._recentMessages.keys().next().value;
          if (oldest !== undefined) this._recentMessages.delete(oldest);
        }

        const isGroup = msg.key.remoteJid?.endsWith('@g.us') || false;

        this.options.onMessage({
          id: msg.key.id || '',
          sender: msg.key.remoteJid || '',
          pn: msg.key.remoteJidAlt || '',
          content: extracted.text,
          timestamp: msg.messageTimestamp as number,
          isGroup,
          quotedText: extracted.quotedText,
        });
      }
    });
  }

  private extractMessageContent(msg: any): { text: string; quotedText?: string } | null {
    const message = msg.message;
    if (!message) return null;

    let text: string | null = null;
    let quotedText: string | undefined;

    // Extended text (reply, link preview) — check first, it has contextInfo
    if (message.extendedTextMessage) {
      text = message.extendedTextMessage.text || null;
      // Extract the quoted message text if the user is replying to a previous message
      const ctx = message.extendedTextMessage.contextInfo;
      if (ctx?.quotedMessage) {
        const qm = ctx.quotedMessage;
        quotedText =
          qm.conversation ||
          qm.extendedTextMessage?.text ||
          qm.imageMessage?.caption ||
          qm.videoMessage?.caption ||
          qm.documentMessage?.caption ||
          undefined;
      }
    }

    // Text message
    if (!text && message.conversation) {
      text = message.conversation;
    }

    // Image with caption
    if (!text && message.imageMessage?.caption) {
      text = `[Image] ${message.imageMessage.caption}`;
    }

    // Video with caption
    if (!text && message.videoMessage?.caption) {
      text = `[Video] ${message.videoMessage.caption}`;
    }

    // Document with caption
    if (!text && message.documentMessage?.caption) {
      text = `[Document] ${message.documentMessage.caption}`;
    }

    // Voice/Audio message
    if (!text && message.audioMessage) {
      text = `[Voice Message]`;
    }

    if (!text) return null;
    return { text, quotedText };
  }

  async sendMessage(to: string, text: string, quoteId?: string): Promise<void> {
    if (!this.sock) {
      throw new Error('Not connected');
    }

    // If a quoteId was provided and we have the original message cached, quote it
    const quoted = quoteId ? this._recentMessages.get(quoteId) : undefined;

    const sent = await this.sock.sendMessage(to, { text }, quoted ? { quoted } : undefined);
    // Track sent message ID to avoid echo when messaging self
    if (sent?.key?.id) {
      this._sentIds.add(sent.key.id);
      // Auto-cleanup after 60s to prevent memory leak
      setTimeout(() => this._sentIds.delete(sent.key.id), 60000);

      // Also cache our own sent messages so they can be quoted later
      if (sent) {
        this._recentMessages.set(sent.key.id, sent);
        if (this._recentMessages.size > 200) {
          const oldest = this._recentMessages.keys().next().value;
          if (oldest !== undefined) this._recentMessages.delete(oldest);
        }
      }
    }
  }

  async disconnect(): Promise<void> {
    if (this.sock) {
      this.sock.end(undefined);
      this.sock = null;
    }
  }
}
