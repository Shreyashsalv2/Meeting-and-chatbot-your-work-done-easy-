"use client";

import { Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { AuthMe } from "@/lib/types";

/**
 * Wraps the app. When Google auth is configured and the user isn't signed in,
 * shows a "Continue with Google" screen; otherwise renders the app. When auth
 * isn't configured, it's a passthrough (existing no-login behavior).
 */
export default function AuthGate({ children }: { children: React.ReactNode }) {
  const [me, setMe] = useState<AuthMe | null>(null);

  useEffect(() => {
    api
      .authMe()
      .then(setMe)
      .catch(() => setMe({ authenticated: false, auth_enabled: false }));
  }, []);

  if (me === null) {
    return (
      <div className="grid h-full place-items-center text-sm text-muted">Loading…</div>
    );
  }

  if (me.auth_enabled && !me.authenticated) {
    return (
      <div className="grid h-full place-items-center bg-canvas px-4">
        <div className="w-full max-w-sm rounded-2xl border border-line bg-card p-8 text-center shadow-sm">
          <span className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-xl bg-gradient-to-br from-brand to-indigo-500 text-white">
            <Sparkles size={22} />
          </span>
          <h1 className="text-xl font-semibold text-ink">Welcome to Fireflies</h1>
          <p className="mt-1.5 text-sm text-muted">
            Sign in to browse meetings, chat with the assistant, and connect your Google
            Calendar.
          </p>
          <a
            href={api.googleLoginUrl()}
            className="mt-6 inline-flex w-full items-center justify-center gap-2 rounded-lg border border-line bg-white px-4 py-2.5 text-sm font-medium text-gray-800 transition hover:bg-gray-50"
          >
            <GoogleG /> Continue with Google
          </a>
          <p className="mt-4 text-xs text-muted">
            You&apos;ll sign in with your Google account (or pick the one you&apos;re already
            using).
          </p>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}

function GoogleG() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden>
      <path
        fill="#EA4335"
        d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"
      />
      <path
        fill="#4285F4"
        d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"
      />
      <path
        fill="#FBBC05"
        d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"
      />
      <path
        fill="#34A853"
        d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"
      />
    </svg>
  );
}
