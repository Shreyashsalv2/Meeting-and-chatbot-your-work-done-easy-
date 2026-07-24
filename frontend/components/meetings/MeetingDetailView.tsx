"use client";

import {
  ArrowLeft,
  Calendar,
  ChevronDown,
  Clock,
  Download,
  Pencil,
  Trash2,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { AvatarStack } from "@/components/ui/Avatar";
import { Button } from "@/components/ui/Button";
import { CenteredSpinner, EmptyState, TagPill } from "@/components/ui/feedback";
import { api } from "@/lib/api";
import { formatDate, formatDuration } from "@/lib/format";
import { useDeleteMeeting, useMeeting } from "@/lib/hooks";

import { EditMeetingModal } from "./EditMeetingModal";
import { MeetingChat } from "./MeetingChat";
import { MediaPlayer } from "./MediaPlayer";
import { SummaryPanel } from "./SummaryPanel";
import { TranscriptPanel } from "./TranscriptPanel";

export default function MeetingDetailView({
  id,
  initialSeek,
}: {
  id: number;
  initialSeek?: number;
}) {
  const router = useRouter();
  const { data: meeting, isLoading, isError } = useMeeting(id);
  const del = useDeleteMeeting();

  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const appliedSeek = useRef(false);
  const exportRef = useRef<HTMLDivElement>(null);

  const duration = meeting
    ? meeting.duration_seconds ||
      (meeting.segments.length
        ? meeting.segments[meeting.segments.length - 1].end_time
        : 0)
    : 0;

  // Apply an incoming ?t= seek once the meeting has loaded.
  useEffect(() => {
    if (meeting && !appliedSeek.current) {
      appliedSeek.current = true;
      if (initialSeek != null && !Number.isNaN(initialSeek)) {
        setCurrentTime(Math.min(initialSeek, duration));
      }
    }
  }, [meeting, initialSeek, duration]);

  // Simulated playback clock (no real audio; timestamps drive everything).
  useEffect(() => {
    if (!isPlaying) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      setCurrentTime((t) => {
        const next = t + dt;
        if (next >= duration) {
          setIsPlaying(false);
          return duration;
        }
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, duration]);

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) {
        setExportOpen(false);
      }
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  if (isLoading) return <CenteredSpinner label="Loading meeting…" />;
  if (isError || !meeting) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-16">
        <EmptyState
          title="Meeting not found"
          description="It may have been deleted."
          action={<Button onClick={() => router.push("/")}>Back to meetings</Button>}
        />
      </div>
    );
  }

  const segments = meeting.segments;
  let activeIndex = -1;
  for (let i = 0; i < segments.length; i++) {
    if (segments[i].start_time <= currentTime + 0.001) activeIndex = i;
    else break;
  }

  const seekTo = (time: number, play = false) => {
    setCurrentTime(Math.max(0, Math.min(time, duration)));
    if (play) setIsPlaying(true);
  };

  function onDelete() {
    if (confirm(`Delete "${meeting!.title}"? This cannot be undone.`)) {
      del.mutate(meeting!.id, { onSuccess: () => router.push("/") });
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-6 md:px-8 lg:flex lg:h-full lg:flex-col">
      <button
        onClick={() => router.push("/")}
        className="mb-4 inline-flex shrink-0 items-center gap-1.5 text-sm text-muted transition hover:text-ink"
      >
        <ArrowLeft size={16} /> Meetings
      </button>

      <div className="mb-6 flex shrink-0 flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold tracking-tight text-ink">
            {meeting.title}
          </h1>
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-sm text-muted">
            <span className="inline-flex items-center gap-1.5">
              <Calendar size={14} /> {formatDate(meeting.meeting_date)}
            </span>
            <span className="inline-flex items-center gap-1.5">
              <Clock size={14} /> {formatDuration(duration)}
            </span>
            <span className="inline-flex items-center gap-2">
              <AvatarStack names={meeting.participants.map((p) => p.name)} size={24} />
              {meeting.participants.length} people
            </span>
          </div>
          {meeting.tags.length > 0 && (
            <div className="mt-2.5 flex flex-wrap gap-1.5">
              {meeting.tags.map((t) => (
                <TagPill key={t.id}>{t.name}</TagPill>
              ))}
            </div>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <Button variant="secondary" size="sm" onClick={() => setEditOpen(true)}>
            <Pencil size={15} /> Edit
          </Button>
          <div ref={exportRef} className="relative">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setExportOpen((o) => !o)}
            >
              <Download size={15} /> Export <ChevronDown size={14} />
            </Button>
            {exportOpen && (
              <div className="absolute right-0 z-10 mt-1 w-40 overflow-hidden rounded-lg border border-line bg-card py-1 shadow-lg">
                <a
                  href={api.exportUrl(meeting.id, "md")}
                  onClick={() => setExportOpen(false)}
                  className="block px-3 py-2 text-sm text-ink hover:bg-panel"
                >
                  Markdown (.md)
                </a>
                <a
                  href={api.exportUrl(meeting.id, "txt")}
                  onClick={() => setExportOpen(false)}
                  className="block px-3 py-2 text-sm text-ink hover:bg-panel"
                >
                  Plain text (.txt)
                </a>
              </div>
            )}
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={onDelete}
            className="text-rose-600 hover:bg-rose-50 dark:hover:bg-rose-500/10"
          >
            <Trash2 size={15} />
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:min-h-0 lg:flex-1 lg:grid-cols-[480px_minmax(0,1fr)]">
        <div className="scroll-slim order-2 lg:order-1 lg:min-h-0 lg:overflow-y-auto">
          <SummaryPanel meeting={meeting} onSeekTopic={(t) => seekTo(t, true)} />
        </div>
        <div className="order-1 flex flex-col gap-4 lg:order-2 lg:min-h-0">
          <div className="lg:shrink-0">
            <MediaPlayer
              duration={duration}
              currentTime={currentTime}
              isPlaying={isPlaying}
              onSeek={(t) => seekTo(t)}
              onPlayPause={() => setIsPlaying((p) => !p)}
            />
          </div>
          <div className="lg:flex lg:min-h-0 lg:flex-1">
            <TranscriptPanel
              segments={segments}
              activeIndex={activeIndex}
              isPlaying={isPlaying}
              onSeek={seekTo}
            />
          </div>
        </div>
      </div>

      <EditMeetingModal
        open={editOpen}
        onClose={() => setEditOpen(false)}
        meeting={meeting}
      />

      <MeetingChat meetingId={meeting.id} onSeek={(t) => seekTo(t, true)} />
    </div>
  );
}
