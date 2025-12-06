import { useReducer, useCallback } from 'react';

// Generic form state management with useReducer
export interface FormState<T = Record<string, unknown>> {
  data: T;
  errors: Record<string, string>;
  isSubmitting: boolean;
  isValid: boolean;
}

type FormAction<T> =
  | { type: 'SET_FIELD'; field: keyof T; value: T[keyof T] }
  | { type: 'SET_ERROR'; field: keyof T; error: string }
  | { type: 'CLEAR_ERROR'; field: keyof T }
  | { type: 'SET_SUBMITTING'; isSubmitting: boolean }
  | { type: 'SET_DATA'; data: Partial<T> }
  | { type: 'RESET' }
  | { type: 'VALIDATE' };

function formReducer<T>(state: FormState<T>, action: FormAction<T>): FormState<T> {
  switch (action.type) {
    case 'SET_FIELD':
      return {
        ...state,
        data: { ...state.data, [action.field]: action.value },
        errors: { ...state.errors, [action.field]: '' }
      };
    
    case 'SET_ERROR':
      return {
        ...state,
        errors: { ...state.errors, [action.field]: action.error },
        isValid: false
      };
    
    case 'CLEAR_ERROR': {
      const { [action.field]: removedError, ...restErrors } = state.errors;
      return {
        ...state,
        errors: restErrors,
        isValid: Object.keys(restErrors).length === 0
      };
    }
    
    case 'SET_SUBMITTING':
      return { ...state, isSubmitting: action.isSubmitting };
    
    case 'SET_DATA':
      return {
        ...state,
        data: { ...state.data, ...action.data }
      };
    
    case 'VALIDATE':
      return {
        ...state,
        isValid: Object.keys(state.errors).length === 0
      };
    
    case 'RESET':
      return {
        data: {} as T,
        errors: {},
        isSubmitting: false,
        isValid: true
      };
    
    default:
      return state;
  }
}

export function useFormState<T = Record<string, unknown>>(initialData?: Partial<T>) {
  const [state, dispatch] = useReducer(formReducer<T>, {
    data: (initialData || {}) as T,
    errors: {},
    isSubmitting: false,
    isValid: true
  });

  const setField = useCallback((field: keyof T, value: T[keyof T]) => {
    dispatch({ type: 'SET_FIELD', field, value });
  }, []);

  const setError = useCallback((field: keyof T, error: string) => {
    dispatch({ type: 'SET_ERROR', field, error });
  }, []);

  const clearError = useCallback((field: keyof T) => {
    dispatch({ type: 'CLEAR_ERROR', field });
  }, []);

  const setSubmitting = useCallback((isSubmitting: boolean) => {
    dispatch({ type: 'SET_SUBMITTING', isSubmitting });
  }, []);

  const setData = useCallback((data: Partial<T>) => {
    dispatch({ type: 'SET_DATA', data });
  }, []);

  const validate = useCallback(() => {
    dispatch({ type: 'VALIDATE' });
  }, []);

  const reset = useCallback(() => {
    dispatch({ type: 'RESET' });
  }, []);

  return {
    ...state,
    setField,
    setError,
    clearError,
    setSubmitting,
    setData,
    validate,
    reset
  };
}