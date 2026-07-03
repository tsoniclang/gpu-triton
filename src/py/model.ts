// Structured Python output model. Lowering decides semantics by building
// this model from GPU IR structure; the printer only decides presentation.
// Expression text is composed by the lowering rows and never re-parsed.

export type PyStatement =
  | { readonly kind: "assign"; readonly target: string; readonly value: string }
  | { readonly kind: "expression"; readonly value: string }
  | { readonly kind: "for"; readonly target: string; readonly iterable: string; readonly body: readonly PyStatement[] }
  | { readonly kind: "if"; readonly condition: string; readonly body: readonly PyStatement[] }
  | { readonly kind: "return"; readonly value?: string }
  | { readonly kind: "blank" };

export interface PyFunction {
  readonly name: string;
  readonly parameters: readonly string[];
  readonly decorators: readonly string[];
  readonly body: readonly PyStatement[];
}

export interface PyModule {
  readonly imports: readonly string[];
  readonly functions: readonly PyFunction[];
}
