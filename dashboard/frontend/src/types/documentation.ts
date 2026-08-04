import { ID } from "./common";

export interface DocSection {
  id: ID;
  title: string;
  order: number;
  content: string;
  tags?: string[];
}

export interface DocIndex {
  sections: { id: ID; title: string; order: number; tags?: string[] }[];
  schemaVersion: string;
}
