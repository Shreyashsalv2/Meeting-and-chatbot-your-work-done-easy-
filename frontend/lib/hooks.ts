"use client";

import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { toast } from "sonner";

import { api, type MeetingQuery } from "./api";
import type {
  ActionItemInput,
  MeetingCreateInput,
  MeetingUpdateInput,
} from "./types";

export type SearchMode = "keyword" | "semantic";

export const keys = {
  meetings: (params?: MeetingQuery) =>
    params ? (["meetings", params] as const) : (["meetings"] as const),
  meeting: (id: number) => ["meeting", id] as const,
  search: (q: string, mode: SearchMode = "keyword") => ["search", mode, q] as const,
};

export function useMeetings(params: MeetingQuery = {}) {
  return useQuery({
    queryKey: keys.meetings(params),
    queryFn: () => api.listMeetings(params),
  });
}

export function useMeeting(id: number) {
  return useQuery({
    queryKey: keys.meeting(id),
    queryFn: () => api.getMeeting(id),
    enabled: Number.isFinite(id),
  });
}

export function useSearch(q: string, mode: SearchMode = "keyword") {
  return useQuery({
    queryKey: keys.search(q, mode),
    queryFn: () => (mode === "semantic" ? api.semanticSearch(q) : api.search(q)),
    enabled: q.trim().length > 0,
  });
}

function useInvalidateLists() {
  const qc = useQueryClient();
  return () => qc.invalidateQueries({ queryKey: ["meetings"] });
}

export function useCreateMeeting() {
  const invalidate = useInvalidateLists();
  return useMutation({
    mutationFn: (body: MeetingCreateInput) => api.createMeeting(body),
    onSuccess: () => {
      invalidate();
      toast.success("Meeting created");
    },
    onError: (e: Error) => toast.error(e.message || "Failed to create meeting"),
  });
}

export function useUploadMeeting() {
  const invalidate = useInvalidateLists();
  return useMutation({
    mutationFn: (form: FormData) => api.uploadMeeting(form),
    onSuccess: () => {
      invalidate();
      toast.success("Transcript uploaded");
    },
    onError: (e: Error) => toast.error(e.message || "Upload failed"),
  });
}

export function useUpdateMeeting(id: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: MeetingUpdateInput) => api.updateMeeting(id, body),
    onSuccess: (data) => {
      qc.setQueryData(keys.meeting(id), data);
      qc.invalidateQueries({ queryKey: ["meetings"] });
      toast.success("Meeting updated");
    },
    onError: (e: Error) => toast.error(e.message || "Failed to update meeting"),
  });
}

export function useDeleteMeeting() {
  const invalidate = useInvalidateLists();
  return useMutation({
    mutationFn: (id: number) => api.deleteMeeting(id),
    onSuccess: () => {
      invalidate();
      toast.success("Meeting deleted");
    },
    onError: (e: Error) => toast.error(e.message || "Failed to delete meeting"),
  });
}

export function useRegenerateSummary(id: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.regenerateSummary(id),
    onSuccess: (data) => {
      qc.setQueryData(keys.meeting(id), data);
      toast.success("Summary regenerated");
    },
    onError: (e: Error) => toast.error(e.message || "Failed to regenerate summary"),
  });
}

export function useAddActionItem(meetingId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ActionItemInput) => api.addActionItem(meetingId, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.meeting(meetingId) });
      toast.success("Action item added");
    },
    onError: (e: Error) => toast.error(e.message || "Failed to add action item"),
  });
}

export function useUpdateActionItem(meetingId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: number; body: Partial<ActionItemInput> }) =>
      api.updateActionItem(id, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.meeting(meetingId) }),
    onError: (e: Error) => toast.error(e.message || "Failed to update action item"),
  });
}

export function useDeleteActionItem(meetingId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.deleteActionItem(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.meeting(meetingId) });
      toast.success("Action item removed");
    },
    onError: (e: Error) => toast.error(e.message || "Failed to remove action item"),
  });
}
