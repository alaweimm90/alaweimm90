const FORM_STORAGE_KEY = 'intake-form-draft';

export interface FormData {
  [key: string]: unknown;
}

export class FormService {
  static saveFormData(formData: FormData): void {
    try {
      localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify(formData));
    } catch (error) {
      console.warn('Failed to save form data to localStorage:', error);
    }
  }

  static loadFormData(): FormData | null {
    try {
      const saved = localStorage.getItem(FORM_STORAGE_KEY);
      return saved ? JSON.parse(saved) : null;
    } catch (error) {
      console.warn('Failed to load form data from localStorage:', error);
      return null;
    }
  }

  static clearFormData(): void {
    try {
      localStorage.removeItem(FORM_STORAGE_KEY);
    } catch (error) {
      console.warn('Failed to clear form data from localStorage:', error);
    }
  }

  static validateEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  static calculatePasswordStrength(password: string): number {
    let strength = 0;
    if (password.length >= 8) strength += 1;
    if (/[a-z]/.test(password)) strength += 1;
    if (/[A-Z]/.test(password)) strength += 1;
    if (/[0-9]/.test(password)) strength += 1;
    if (/[^A-Za-z0-9]/.test(password)) strength += 1;
    return Math.min(strength, 4);
  }
}