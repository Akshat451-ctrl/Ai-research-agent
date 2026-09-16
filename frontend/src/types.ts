// Mirrors the Pydantic response models in app/api.py. Keep in sync with
// ResearchResponse / VerificationOut / ClaimOut / SourceOut there.

export interface SourceOut {
  source: string;
  title: string | null;
}

export interface ClaimOut {
  claim: string;
  verdict: "supported" | "contradicted" | "unsupported";
  explanation: string;
}

export interface VerificationOut {
  supported: number;
  total: number;
  revisions: number;
  unresolved: ClaimOut[];
}

export interface ResearchResponse {
  question: string;
  report: string;
  verification: VerificationOut | null;
  sources: SourceOut[];
}
