export interface AcademicProfile {
  id: string;
  name: string;
  title: string;
  institution: string;
  email: string;
  phone?: string;
  bio: string;
  photo: string;
  socialLinks: SocialLink[];
  expertise: string[];
  education: Education[];
  awards: Award[];
}

export interface SocialLink {
  platform: 'linkedin' | 'google-scholar' | 'orcid' | 'email' | 'twitter';
  url: string;
  label: string;
}

export interface Education {
  degree: string;
  institution: string;
  year: string;
  field?: string;
}

export interface Award {
  title: string;
  organization: string;
  year: string;
  description?: string;
}

export interface Publication {
  id: string;
  title: string;
  abstract: string;
  authors: Author[];
  journal: string;
  date: Date;
  doi: string;
  url: string;
  pdfUrl: string;
  citations: number;
  type: PublicationType;
  status: PublicationStatus;
  keywords: string[];
  volume?: string;
  issue?: string;
  pages?: string;
  publisher?: string;
}

export interface Author {
  name: string;
  affiliation?: string;
  orcid?: string;
  isCorresponding?: boolean;
}

export interface ResearchProject {
  id: string;
  title: string;
  description: string;
  category: ResearchCategory;
  startDate: Date;
  endDate?: Date;
  status: ProjectStatus;
  funding?: FundingInfo;
  collaborators: Collaborator[];
  publications: string[];
  images: string[];
  outcomes: string[];
  technologies: string[];
}

export interface FundingInfo {
  agency: string;
  grantNumber: string;
  amount: number;
  currency: string;
  startDate: Date;
  endDate: Date;
}

export interface Collaborator {
  name: string;
  institution: string;
  role: string;
  email?: string;
}

export interface Course {
  id: string;
  title: string;
  code: string;
  level: CourseLevel;
  institution: string;
  semester: string;
  year: string;
  description: string;
  syllabus: string;
  materials: CourseMaterial[];
  enrollment: number;
  credits: number;
  schedule: CourseSchedule;
}

export interface CourseMaterial {
  type: 'syllabus' | 'lecture' | 'assignment' | 'reading' | 'video';
  title: string;
  url: string;
  description?: string;
  datePosted: Date;
}

export interface CourseSchedule {
  daysOfWeek: string[];
  startTime: string;
  endTime: string;
  location: string;
}

export interface TeachingExperience {
  institution: string;
  position: string;
  startDate: Date;
  endDate?: Date;
  courses: string[];
  responsibilities: string[];
}

export interface Presentation {
  id: string;
  title: string;
  event: string;
  location: string;
  date: Date;
  type: PresentationType;
  url?: string;
  materials?: string[];
}

export interface CitationMetrics {
  totalCitations: number;
  hIndex: number;
  i10Index: number;
  recentCitations: number;
  topPapers: TopPaper[];
}

export interface TopPaper {
  publicationId: string;
  title: string;
  citations: number;
  year: number;
}

export interface AcademicEvent {
  id: string;
  title: string;
  type: EventType;
  startDate: Date;
  endDate?: Date;
  location: string;
  description: string;
  url?: string;
  organizer?: string;
}

// Type Definitions
export type PublicationType =
  | 'journal-article'
  | 'conference-paper'
  | 'book-chapter'
  | 'patent'
  | 'technical-report'
  | 'preprint'
  | 'thesis'
  | 'book';

export type PublicationStatus =
  | 'published'
  | 'in-review'
  | 'preprint'
  | 'draft'
  | 'accepted';

export type ResearchCategory =
  | 'materials-science'
  | 'computational-physics'
  | 'nanotechnology'
  | 'quantum-mechanics'
  | 'energy-research'
  | 'biophysics'
  | 'theoretical-physics'
  | 'experimental-physics';

export type ProjectStatus =
  | 'active'
  | 'completed'
  | 'planned'
  | 'on-hold';

export type CourseLevel =
  | 'undergraduate'
  | 'graduate'
  | 'seminar'
  | 'workshop'
  | 'postdoctoral';

export type PresentationType =
  | 'invited-talk'
  | 'conference-talk'
  | 'poster'
  | 'keynote'
  | 'workshop';

export type EventType =
  | 'conference'
  | 'workshop'
  | 'seminar'
  | 'colloquium'
  | 'defense'
  | 'committee-meeting';

// API Response Types
export interface PublicationsResponse {
  publications: Publication[];
  total: number;
  page: number;
  pageSize: number;
}

export interface ResearchResponse {
  projects: ResearchProject[];
  total: number;
}

export interface CoursesResponse {
  courses: Course[];
  teaching: TeachingExperience[];
}

export interface MetricsResponse {
  metrics: CitationMetrics;
  lastUpdated: Date;
}

// Search and Filter Types
export interface SearchFilters {
  type?: PublicationType[];
  category?: ResearchCategory[];
  dateRange?: {
    start: Date;
    end: Date;
  };
  author?: string;
  keyword?: string;
  journal?: string;
}

export interface SortOptions {
  field: 'date' | 'citations' | 'title' | 'journal';
  direction: 'asc' | 'desc';
}

// Component Props Types
export interface PublicationCardProps {
  publication: Publication;
  onViewDetails: (id: string) => void;
  onDownloadPDF: (url: string) => void;
  compact?: boolean;
  showAbstract?: boolean;
}

export interface ResearchProjectCardProps {
  project: ResearchProject;
  onViewDetails: (id: string) => void;
  showStatus?: boolean;
}

export interface CourseCardProps {
  course: Course;
  onViewDetails: (id: string) => void;
  showSchedule?: boolean;
}

export interface TimelineItemProps {
  date: Date;
  title: string;
  description: string;
  type: 'education' | 'career' | 'award' | 'publication';
  institution?: string;
}
