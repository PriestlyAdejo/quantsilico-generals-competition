export type CapabilityState =
  | "IMPLEMENTED"
  | "INTENTIONALLY_EMPTY"
  | "CAPABILITY_DISABLED"
  | "NOT_APPLICABLE";

export class ApiError extends Error {
  readonly code: string;
  readonly status?: number;

  constructor(code: string, message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

export class CapabilityDisabledError extends Error {
  readonly state: CapabilityState = "CAPABILITY_DISABLED";
  readonly method: string;

  constructor(method: string, reason: string) {
    super(reason);
    this.name = "CapabilityDisabledError";
    this.method = method;
  }
}
