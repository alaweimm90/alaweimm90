// Utility types - Generic TypeScript utilities
// These types provide common type transformations and utilities

// Make specific properties optional
export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

// Make specific properties required
export type RequiredKeys<T, K extends keyof T> = T & Required<Pick<T, K>>;

// Deep partial - makes all properties and nested properties optional
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

// Deep required - makes all properties and nested properties required
export type DeepRequired<T> = {
  [P in keyof T]-?: T[P] extends object ? DeepRequired<T[P]> : T[P];
};

// Non-empty array type
export type NonEmptyArray<T> = [T, ...T[]];

// Extract array element type
export type ArrayElement<T> = T extends readonly (infer U)[] ? U : never;

// Extract promise resolution type
export type Awaited<T> = T extends PromiseLike<infer U> ? U : T;

// Prettify - makes complex types more readable in IDE
export type Prettify<T> = {
  [K in keyof T]: T[K];
} & {};

// Exact type - prevents extra properties
export type Exact<T, U extends T> = T & Record<Exclude<keyof U, keyof T>, never>;

// Mutable - removes readonly modifiers
export type Mutable<T> = {
  -readonly [P in keyof T]: T[P];
};

// Merge two types with second overriding first
export type Merge<T, U> = Omit<T, keyof U> & U;

// Override specific properties in a type
export type Override<T, U> = Omit<T, keyof U> & U;

// Extract keys that have specific value type
export type KeysOfType<T, U> = {
  [K in keyof T]: T[K] extends U ? K : never;
}[keyof T];

// Extract properties that have specific value type
export type PropertiesOfType<T, U> = Pick<T, KeysOfType<T, U>>;

// String literal utilities
export type Uppercase<S extends string> = intrinsic;
export type Lowercase<S extends string> = intrinsic;
export type Capitalize<S extends string> = intrinsic;
export type Uncapitalize<S extends string> = intrinsic;

// Object path type for nested property access
export type Paths<T> = T extends object
  ? {
      [K in keyof T]: K extends string
        ? T[K] extends object
          ? `${K}` | `${K}.${Paths<T[K]>}`
          : `${K}`
        : never;
    }[keyof T]
  : never;

// Get type at path
export type PathValue<T, P extends Paths<T>> = P extends `${infer K}.${infer Rest}`
  ? K extends keyof T
    ? Rest extends Paths<T[K]>
      ? PathValue<T[K], Rest>
      : never
    : never
  : P extends keyof T
  ? T[P]
  : never;

// Function type utilities
export type Parameters<T extends (...args: unknown[]) => unknown> = T extends (...args: infer P) => unknown ? P : never;
export type ReturnType<T extends (...args: unknown[]) => unknown> = T extends (...args: unknown[]) => infer R ? R : unknown;

// Async function utilities
export type AsyncReturnType<T extends (...args: unknown[]) => Promise<unknown>> =
  T extends (...args: unknown[]) => Promise<infer R> ? R : never;

// Class utilities
export type ConstructorParameters<T extends abstract new (...args: unknown[]) => unknown> =
  T extends abstract new (...args: infer P) => unknown ? P : never;

export type InstanceType<T extends abstract new (...args: unknown[]) => unknown> =
  T extends abstract new (...args: unknown[]) => infer R ? R : unknown;

// Tuple utilities
export type Head<T extends readonly unknown[]> = T extends readonly [infer H, ...unknown[]] ? H : never;
export type Tail<T extends readonly unknown[]> = T extends readonly [unknown, ...infer R] ? R : [];
export type Length<T extends readonly unknown[]> = T['length'];

// Union utilities
export type UnionToIntersection<U> = (U extends unknown ? (arg: U) => unknown : never) extends 
  (arg: infer I) => unknown ? I : never;

export type UnionToTuple<T> = UnionToIntersection<
  T extends unknown ? (t: T) => T : never
> extends (_: unknown) => infer W
  ? [...UnionToTuple<Exclude<T, W>>, W]
  : [];

// Conditional type utilities
export type If<C extends boolean, T, F> = C extends true ? T : F;
export type Not<C extends boolean> = C extends true ? false : true;
export type And<A extends boolean, B extends boolean> = A extends true 
  ? B extends true 
    ? true 
    : false 
  : false;
export type Or<A extends boolean, B extends boolean> = A extends true 
  ? true 
  : B extends true 
    ? true 
    : false;

// Brand types for nominal typing
export declare const brand: unique symbol;
export type Brand<T, B> = T & { [brand]: B };

// Common branded types
export type UserId = Brand<string, 'UserId'>;
export type Email = Brand<string, 'Email'>;
export type PhoneNumber = Brand<string, 'PhoneNumber'>;
export type Timestamp = Brand<number, 'Timestamp'>;
export type Currency = Brand<number, 'Currency'>;

// Validation result types
export type ValidationResult<T> = 
  | { success: true; data: T }
  | { success: false; error: string; details?: Record<string, unknown> };

// Either type for error handling
export type Either<L, R> = 
  | { type: 'left'; value: L }
  | { type: 'right'; value: R };

// Maybe type for nullable values
export type Maybe<T> = T | null | undefined;

// Result type for operations that can fail
export type Result<T, E = Error> = 
  | { success: true; value: T }
  | { success: false; error: E };

// State machine types
export type StateMachine<S, E> = {
  state: S;
  transition: (event: E) => StateMachine<S, E>;
};

// Event emitter types
export type EventMap = Record<string, unknown>;
export type EventListener<T> = (event: T) => void;
export type Unsubscribe = () => void;