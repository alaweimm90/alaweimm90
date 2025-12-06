import { useState, useEffect, useCallback } from 'react';
import { i18n, t, formatNumber, formatDate, formatRelativeTime, SupportedLanguage, SUPPORTED_LANGUAGES } from '@/lib/i18n';
import { trackQuantumEvents } from '@/lib/analytics';
import { logger } from '@/lib/logger';

interface I18nState {
  currentLanguage: SupportedLanguage;
  isLoading: boolean;
  isRTL: boolean;
  availableLanguages: typeof SUPPORTED_LANGUAGES;
}

interface I18nActions {
  setLanguage: (language: SupportedLanguage) => Promise<void>;
  translate: (key: string, fallback?: string) => string;
  formatNumber: (number: number, options?: Intl.NumberFormatOptions) => string;
  formatDate: (date: Date, options?: Intl.DateTimeFormatOptions) => string;
  formatRelativeTime: (date: Date) => string;
}

export const useI18n = (): I18nState & I18nActions => {
  const [state, setState] = useState<I18nState>({
    currentLanguage: i18n.getCurrentLanguage(),
    isLoading: false,
    isRTL: i18n.isRTL(),
    availableLanguages: SUPPORTED_LANGUAGES
  });

  // Handle language changes
  useEffect(() => {
    const handleLanguageChange = (event: CustomEvent) => {
      setState(prev => ({
        ...prev,
        currentLanguage: event.detail.language,
        isRTL: i18n.isRTL()
      }));
    };

    window.addEventListener('languageChanged', handleLanguageChange as EventListener);
    
    return () => {
      window.removeEventListener('languageChanged', handleLanguageChange as EventListener);
    };
  }, []);

  // Set language with loading state
  const setLanguage = useCallback(async (language: SupportedLanguage): Promise<void> => {
    setState(prev => ({ ...prev, isLoading: true }));
    
    try {
      await i18n.setLanguage(language);
      trackQuantumEvents.preferenceChange('language_selected', language, state.currentLanguage);
    } catch (error) {
      logger.error('Failed to set language', { error, language });
      trackQuantumEvents.errorBoundary(
        'Language change failed',
        (error as Error).stack || 'No stack trace',
        'i18n'
      );
    } finally {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  }, [state.currentLanguage]);

  return {
    ...state,
    setLanguage,
    translate: t,
    formatNumber,
    formatDate,
    formatRelativeTime
  };
};

// Hook for specific quantum terminology translations
export const useQuantumTerms = () => {
  const { translate } = useI18n();

  return {
    qubit: translate('quantum.qubit'),
    qubits: translate('quantum.qubits'),
    gate: translate('quantum.gate'),
    gates: translate('quantum.gates'),
    circuit: translate('quantum.circuit'),
    circuits: translate('quantum.circuits'),
    state: translate('quantum.state'),
    superposition: translate('quantum.superposition'),
    entanglement: translate('quantum.entanglement'),
    measurement: translate('quantum.measurement'),
    amplitude: translate('quantum.amplitude'),
    phase: translate('quantum.phase'),
    probability: translate('quantum.probability'),
    fidelity: translate('quantum.fidelity'),
    blochSphere: translate('quantum.bloch_sphere'),
    hadamard: translate('quantum.hadamard'),
    pauliX: translate('quantum.pauli_x'),
    pauliY: translate('quantum.pauli_y'),
    pauliZ: translate('quantum.pauli_z'),
    cnot: translate('quantum.cnot'),
    rotation: translate('quantum.rotation'),
    training: translate('quantum.training'),
    simulation: translate('quantum.simulation'),
    optimization: translate('quantum.optimization'),
    gradient: translate('quantum.gradient'),
    lossFunction: translate('quantum.loss_function'),
    accuracy: translate('quantum.accuracy'),
    epoch: translate('quantum.epoch'),
    learningRate: translate('quantum.learning_rate'),
    vqc: translate('quantum.vqc'),
    qml: translate('quantum.qml')
  };
};

// Hook for navigation translations
export const useNavigation = () => {
  const { translate } = useI18n();

  return {
    home: translate('navigation.home'),
    circuits: translate('navigation.circuits'),
    training: translate('navigation.training'),
    visualization: translate('navigation.visualization'),
    docs: translate('navigation.docs'),
    settings: translate('navigation.settings'),
    help: translate('navigation.help')
  };
};

// Hook for common UI translations
export const useCommonTranslations = () => {
  const { translate } = useI18n();

  return {
    loading: translate('common.loading'),
    error: translate('common.error'),
    success: translate('common.success'),
    cancel: translate('common.cancel'),
    save: translate('common.save'),
    delete: translate('common.delete'),
    edit: translate('common.edit'),
    close: translate('common.close'),
    next: translate('common.next'),
    previous: translate('common.previous'),
    submit: translate('common.submit'),
    reset: translate('common.reset'),
    clear: translate('common.clear'),
    copy: translate('common.copy'),
    share: translate('common.share')
  };
};

// Hook for accessibility translations
export const useAccessibilityTranslations = () => {
  const { translate } = useI18n();

  return {
    skipToMain: translate('accessibility.skip_to_main'),
    menuButton: translate('accessibility.menu_button'),
    closeMenu: translate('accessibility.close_menu'),
    expandSection: translate('accessibility.expand_section'),
    collapseSection: translate('accessibility.collapse_section'),
    loadingContent: translate('accessibility.loading_content'),
    errorOccurred: translate('accessibility.error_occurred'),
    quantumStateDescription: translate('accessibility.quantum_state_description'),
    circuitGateDescription: translate('accessibility.circuit_gate_description'),
    trainingProgressDescription: translate('accessibility.training_progress_description')
  };
};

// Hook for error message translations
export const useErrorTranslations = () => {
  const { translate } = useI18n();

  return {
    invalidCircuit: translate('errors.invalid_circuit'),
    compilationFailed: translate('errors.compilation_failed'),
    simulationError: translate('errors.simulation_error'),
    trainingFailed: translate('errors.training_failed'),
    networkError: translate('errors.network_error'),
    fileNotFound: translate('errors.file_not_found'),
    permissionDenied: translate('errors.permission_denied'),
    invalidInput: translate('errors.invalid_input'),
    quotaExceeded: translate('errors.quota_exceeded'),
    serviceUnavailable: translate('errors.service_unavailable')
  };
};

// Hook for tutorial translations
export const useTutorialTranslations = () => {
  const { translate } = useI18n();

  return {
    welcomeTitle: translate('tutorial.welcome_title'),
    welcomeDescription: translate('tutorial.welcome_description'),
    circuitBasicsTitle: translate('tutorial.circuit_basics_title'),
    circuitBasicsDescription: translate('tutorial.circuit_basics_description'),
    quantumGatesTitle: translate('tutorial.quantum_gates_title'),
    quantumGatesDescription: translate('tutorial.quantum_gates_description'),
    measurementTitle: translate('tutorial.measurement_title'),
    measurementDescription: translate('tutorial.measurement_description'),
    machineLearningTitle: translate('tutorial.machine_learning_title'),
    machineLearningDescription: translate('tutorial.machine_learning_description'),
    nextStep: translate('tutorial.next_step'),
    skipTutorial: translate('tutorial.skip_tutorial'),
    restartTutorial: translate('tutorial.restart_tutorial')
  };
};

// Hook for formatted numbers in current locale
export const useFormattedNumbers = () => {
  const { formatNumber, currentLanguage } = useI18n();

  const formatPercentage = useCallback((value: number, decimals: number = 1) => {
    return formatNumber(value, {
      style: 'percent',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  }, [formatNumber]);

  const formatCurrency = useCallback((value: number, currency: string = 'USD') => {
    return formatNumber(value, {
      style: 'currency',
      currency
    });
  }, [formatNumber]);

  const formatDecimal = useCallback((value: number, decimals: number = 2) => {
    return formatNumber(value, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  }, [formatNumber]);

  const formatInteger = useCallback((value: number) => {
    return formatNumber(value, {
      maximumFractionDigits: 0
    });
  }, [formatNumber]);

  const formatScientific = useCallback((value: number, decimals: number = 2) => {
    return formatNumber(value, {
      notation: 'scientific',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  }, [formatNumber]);

  return {
    formatNumber,
    formatPercentage,
    formatCurrency,
    formatDecimal,
    formatInteger,
    formatScientific,
    currentLanguage
  };
};

// Hook for quantum-specific number formatting
export const useQuantumNumbers = () => {
  const { formatNumber } = useFormattedNumbers();

  const formatProbability = useCallback((value: number) => {
    return formatNumber(value, {
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    });
  }, [formatNumber]);

  const formatAmplitude = useCallback((value: number) => {
    return formatNumber(value, {
      minimumFractionDigits: 3,
      maximumFractionDigits: 6,
      signDisplay: 'always'
    });
  }, [formatNumber]);

  const formatPhase = useCallback((value: number) => {
    return formatNumber(value, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }) + ' rad';
  }, [formatNumber]);

  const formatFidelity = useCallback((value: number) => {
    return formatNumber(value, {
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 3
    });
  }, [formatNumber]);

  const formatAngle = useCallback((value: number, unit: 'rad' | 'deg' = 'rad') => {
    const formatted = formatNumber(value, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    });
    return `${formatted} ${unit}`;
  }, [formatNumber]);

  return {
    formatProbability,
    formatAmplitude,
    formatPhase,
    formatFidelity,
    formatAngle
  };
};