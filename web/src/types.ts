export type LawRegion = "亚洲" | "欧洲" | "中东" | "北美洲" | "东南亚";

export type LawCategory =
  | "言论"
  | "公共秩序"
  | "网络监管"
  | "道德规范"
  | "个人安全";

export type ReviewStatus = "verified" | "needs-source" | "contested";

export interface LawEntry {
  id: string;
  title: string;
  country: string;
  region: LawRegion;
  category: LawCategory;
  year: string;
  severity: number;
  summary: string;
  impact: string;
  whyItFeelsScary: string;
  researchLead: string;
  reviewStatus: ReviewStatus;
}

