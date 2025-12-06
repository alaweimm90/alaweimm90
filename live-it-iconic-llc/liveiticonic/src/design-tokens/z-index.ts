// ============================================================================
// LIVE IT ICONIC - Z-INDEX DESIGN TOKENS
// Stacking Context & Layer Management
// ============================================================================

export const zIndex = {
  // ========================================================================
  // BASE LAYERS
  // ========================================================================
  hide: -1, // Hidden elements
  base: 0, // Default/body content

  // ========================================================================
  // INTERACTIVE LAYERS
  // ========================================================================
  docked: 10, // Sticky, pinned elements
  dropdown: 1000, // Dropdown menus
  sticky: 1020, // Sticky headers
  banner: 1030, // Banner notifications
  overlay: 1040, // Overlay/backdrop
  modal: 1050, // Modal dialogs
  popover: 1060, // Popovers, tooltips
  skipLink: 1070, // Skip navigation link
  toast: 1080, // Toast notifications
  tooltip: 1090, // Floating tooltips

  // ========================================================================
  // LEGACY ALIASES - Backward Compatibility
  // ========================================================================
  fixed: 1200, // Fixed positioned elements
  absolute: 1300, // Absolutely positioned elements
} as const;

// ============================================================================
// Z-INDEX CATEGORIES - Organizational Structure
// ============================================================================
export const zIndexCategories = {
  // Hidden and base elements
  hidden: {
    hide: zIndex.hide,
    base: zIndex.base,
  },

  // Floating elements
  floating: {
    docked: zIndex.docked,
    dropdown: zIndex.dropdown,
    sticky: zIndex.sticky,
    banner: zIndex.banner,
  },

  // Overlay and modal layers
  overlay: {
    overlay: zIndex.overlay,
    modal: zIndex.modal,
    popover: zIndex.popover,
  },

  // Top-level elements
  topmost: {
    skipLink: zIndex.skipLink,
    toast: zIndex.toast,
    tooltip: zIndex.tooltip,
  },
} as const;

// ============================================================================
// Z-INDEX PRESETS - Component Specific
// ============================================================================
export const zIndexPresets = {
  // Navigation
  navigation: {
    sticky: zIndex.sticky,
    drawer: zIndex.modal,
    megaMenu: zIndex.popover,
  },

  // Modal/Dialog
  modal: {
    backdrop: zIndex.overlay,
    content: zIndex.modal,
    closeButton: zIndex.modal + 1,
  },

  // Notifications
  notification: {
    toast: zIndex.toast,
    banner: zIndex.banner,
    badge: zIndex.toast - 1,
  },

  // Floating elements
  floating: {
    dropdown: zIndex.dropdown,
    popover: zIndex.popover,
    tooltip: zIndex.tooltip,
  },

  // Dropdowns and menus
  dropdown: {
    backdrop: zIndex.overlay - 10,
    menu: zIndex.dropdown,
    submenu: zIndex.dropdown + 1,
  },

  // Popovers and tooltips
  popover: {
    backdrop: zIndex.overlay - 5,
    content: zIndex.popover,
    arrow: zIndex.popover,
  },

  // Sticky elements
  sticky: {
    header: zIndex.sticky,
    footer: zIndex.sticky - 1,
    sidebar: zIndex.sticky - 2,
  },

  // Overlays and backdrops
  backdrop: {
    dark: zIndex.overlay,
    light: zIndex.overlay - 1,
  },

  // Floating action button
  fab: {
    default: zIndex.popover - 10,
    hover: zIndex.popover - 9,
    active: zIndex.popover - 8,
  },
} as const;

// ============================================================================
// Z-INDEX DOCUMENTATION & REFERENCE
// ============================================================================
export const zIndexReference = {
  '-1': 'Hidden elements, backups',
  '0': 'Base/default content, body',
  '10': 'Docked/sticky elements',
  '1000': 'Dropdown menus',
  '1020': 'Sticky headers, navigation',
  '1030': 'Banner notifications',
  '1040': 'Overlay backdrops',
  '1050': 'Modal dialogs, main content',
  '1060': 'Popovers, secondary modals',
  '1070': 'Skip navigation links',
  '1080': 'Toast notifications, top alerts',
  '1090': 'Floating tooltips, absolute topmost',
  '1200': 'Fixed positioned elements',
  '1300': 'Absolutely positioned elements',
} as const;

// ============================================================================
// TYPE EXPORTS
// ============================================================================
export type ZIndex = typeof zIndex;
export type ZIndexKey = keyof typeof zIndex;
export type ZIndexCategory = keyof typeof zIndexCategories;
export type ZIndexPresetKey = keyof typeof zIndexPresets;

// ============================================================================
// Z-INDEX UTILITY FUNCTIONS
// ============================================================================
export const getZIndex = (key: ZIndexKey): number => {
  return zIndex[key];
};

/**
 * Gets the z-index for a specific component
 */
export const getComponentZIndex = (component: string): number => {
  const componentMap: Record<string, ZIndexKey> = {
    modal: 'modal',
    dropdown: 'dropdown',
    tooltip: 'tooltip',
    toast: 'toast',
    popover: 'popover',
    sticky: 'sticky',
    overlay: 'overlay',
    banner: 'banner',
    navigation: 'sticky',
  };

  const key = componentMap[component];
  return key ? getZIndex(key) : zIndex.base;
};

/**
 * Creates a layered z-index for stacked components
 * @param baseKey - The base z-index key
 * @param layerOffset - Number to add for each layer (default: 1)
 * @param layers - Number of layers (default: 1)
 */
export const createLayeredZIndex = (
  baseKey: ZIndexKey,
  layerOffset: number = 1,
  layers: number = 1
): number[] => {
  const baseValue = getZIndex(baseKey);
  const result: number[] = [baseValue];

  for (let i = 1; i < layers; i++) {
    result.push(baseValue + i * layerOffset);
  }

  return result;
};

/**
 * Validates that z-index values don't conflict
 */
export const validateZIndexStack = (components: Record<string, number>): boolean => {
  const values = Object.values(components);
  const uniqueValues = new Set(values);
  return values.length === uniqueValues.size;
};

/**
 * Gets all z-index values sorted from lowest to highest
 */
export const sortedZIndexes = (): Array<[string, number]> => {
  return (Object.entries(zIndex) as Array<[string, number]>)
    .sort((a, b) => a[1] - b[1]);
};

// ============================================================================
// CSS HELPER FUNCTIONS
// ============================================================================
/**
 * CSS-in-JS object for z-index
 */
export const createZIndexStyle = (level: ZIndexKey): object => ({
  zIndex: getZIndex(level),
});

/**
 * Creates a modal backdrop with appropriate z-index
 */
export const createModalBackdrop = (): object => ({
  zIndex: zIndex.overlay,
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundColor: 'rgba(0, 0, 0, 0.5)',
});

/**
 * Creates a modal content container with appropriate z-index
 */
export const createModalContent = (): object => ({
  zIndex: zIndex.modal,
  position: 'fixed',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
});

/**
 * Creates a sticky header with appropriate z-index
 */
export const createStickyHeader = (): object => ({
  zIndex: zIndex.sticky,
  position: 'sticky',
  top: 0,
});

/**
 * Creates a dropdown menu with appropriate z-index
 */
export const createDropdown = (): object => ({
  zIndex: zIndex.dropdown,
  position: 'absolute',
});

/**
 * Creates a toast notification with appropriate z-index
 */
export const createToast = (): object => ({
  zIndex: zIndex.toast,
  position: 'fixed',
});
