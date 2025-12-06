/**
 * Utility function for combining class names
 * Similar to clsx/classnames but lightweight
 */

export type ClassValue = ClassArray | ClassDictionary | string | number | null | boolean | undefined;
export type ClassDictionary = Record<string, any>;
export type ClassArray = ClassValue[];

function toVal(mix: ClassValue): string {
  let k: any;
  let y: any;
  let str = '';

  if (typeof mix === 'string' || typeof mix === 'number') {
    str += mix;
  } else if (typeof mix === 'object') {
    if (Array.isArray(mix)) {
      for (k = 0; k < mix.length; k++) {
        if (mix[k]) {
          if ((y = toVal(mix[k]))) {
            if (str) str += ' ';
            str += y;
          }
        }
      }
    } else {
      for (k in mix) {
        if (mix[k]) {
          if (str) str += ' ';
          str += k;
        }
      }
    }
  }

  return str;
}

/**
 * Combines class names into a single string
 * @param inputs - Class values to combine
 * @returns Combined class string
 */
export function cn(...inputs: ClassValue[]): string {
  let i = 0;
  let tmp: any;
  let x: any;
  let str = '';

  while (i < inputs.length) {
    if ((tmp = inputs[i++])) {
      if ((x = toVal(tmp))) {
        if (str) str += ' ';
        str += x;
      }
    }
  }

  return str;
}

/**
 * Conditional class names helper
 * @param condition - Condition to check
 * @param trueClass - Class to apply if true
 * @param falseClass - Class to apply if false (optional)
 * @returns Class string
 */
export function conditionalClass(
  condition: boolean,
  trueClass: string,
  falseClass: string = ''
): string {
  return condition ? trueClass : falseClass;
}

/**
 * Merge Tailwind classes with proper precedence
 * Later classes override earlier ones
 */
export function twMerge(...classes: ClassValue[]): string {
  const classString = cn(...classes);
  const classList = classString.split(' ');
  const classMap = new Map<string, string>();

  classList.forEach((cls) => {
    if (!cls) return;

    // Extract the base class (e.g., 'bg' from 'bg-red-500')
    const match = cls.match(/^([a-z]+(?:-[a-z]+)*)?/);
    if (match) {
      const baseClass = match[0];
      // Handle special cases where we want to keep multiple classes
      const keepMultiple = [
        'animate',
        'transition',
        'transform',
        'filter',
        'backdrop',
      ];

      if (keepMultiple.some(prefix => cls.startsWith(prefix))) {
        // For these, we keep all classes
        classMap.set(cls, cls);
      } else {
        // For others, we override
        classMap.set(baseClass, cls);
      }
    }
  });

  return Array.from(classMap.values()).join(' ');
}