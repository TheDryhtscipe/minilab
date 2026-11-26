import Lean

/-!
# The guard for model-written Lean (build plan doc-5wh, Tasks 9 and 11)

The Lean runner (`minilab.lean.runner.check_lean`) puts `import Minilab.Guard`
before every file it checks. Importing this module does two things.

**Code generation is off.** The `initialize` below runs when the file's
imports load, before the file's first command. It replaces Lean's compiler
with one that gives no code, so no declaration the file makes gets code.
Lean runs code only through compiled code (`#eval`, `run_cmd`, `eval%`,
`decide +native`, an `aesop` rule, a tactic or term elaborator that evaluates
a term), so nothing the file writes can run while Lean checks it. Code that
was compiled before, Lean's and Mathlib's, still runs as usual: tactics,
`simp`, `norm_num`, and the commands below. A computable `def` still
elaborates and type-checks; it just has no code.

**The checks are commands, not text.** The checks used to be Lean text
appended to the file and run with `#eval`. With code generation off,
`#eval` cannot run. So each check is a command defined here, compiled when
this module is built, and given its inputs as literals:

- `#minilab_axioms`: the axioms of every declaration in the file.
- `#minilab_command_check "marker" "source"`: parse a formalisation held as
  a string and list what it may not use.
- `#minilab_probe "marker" arity "opens"`: the binders and conclusion of
  `conjecture`, and the file's definitions, as Lean prints them.
- `#minilab_vacuity "nonce" arity "witness JSON"`: declare the goals the
  vacuity linter proves.

Every result is an information message that starts with a marker the
caller chose for this run.
-/

open Lean Meta Elab Command

namespace Minilab.Guard

/-- In place of Lean's compiler: give no code, and mark each declaration
`noncomputable`. Code that needs a declaration's code then does not run,
often with no message: `#eval` reports nothing and does nothing, and an
Aesop rule clause that names the file's tactic leaves only the warning that
the declaration uses `sorry`. -/
initialize
  compileDeclsRef.set (fun names _ => names.forM fun n => modifyEnv (addNoncomputable · n))

/-- An information message `marker ++ tag ++ " " ++ json`. -/
def say {m : Type → Type} [Monad m] [MonadLog m] [AddMessageContext m] [MonadOptions m]
    (marker tag : String) (j : Json) : m Unit :=
  logInfo (MessageData.ofFormat (Std.Format.text (marker ++ tag ++ " " ++ j.compress)))

/-- The name a binder was written with, or `""` for a binder with no name
(`_`, `[Foo]`), which Lean names with macro scopes. -/
def binderName (n : Name) : String :=
  if n.hasMacroScopes then "" else n.toString

-- ## The axiom check

syntax (name := axiomsCmd) "#minilab_axioms" : command

/-- For every constant the file declares (not the imported ones): its name
with any private prefix removed, whether it is a theorem, whether the name
is internal, and the axioms it depends on. -/
@[command_elab axiomsCmd] def elabAxioms : CommandElab := fun _ => do
  let env ← getEnv
  let mut out : Array Json := #[]
  for (name, info) in env.constants.map₂.toList do
    let found ← collectAxioms name
    let user := (privateToUserName? name).getD name
    let isTheorem := match info with
      | ConstantInfo.thmInfo _ => true
      | _ => false
    out := out.push (Json.mkObj [
      ("name", toJson user.toString),
      ("theorem", toJson isTheorem),
      ("internal", toJson user.isInternal),
      ("axioms", toJson (found.map Name.toString))])
  logInfo (MessageData.ofFormat (Std.Format.text ("MINILAB_AXIOMS " ++ (Json.arr out).compress)))

-- ## The command check

/-- The top-level commands a formalisation may use. `open … in` is allowed
around an allowed command, but not around a command that opens or closes a
scope (`open`, `namespace`, `section`, `end`). -/
def allowedCommands : List Name :=
  [``Lean.Parser.Command.open, ``Lean.Parser.Command.namespace,
   ``Lean.Parser.Command.section, ``Lean.Parser.Command.end,
   ``Lean.Parser.Command.universe, ``Lean.Parser.Command.declaration, `lemma]

/-- The kinds of `declaration` a formalisation may use: no `instance`,
`axiom`, `opaque`, or `example`. -/
def allowedDeclarations : List Name :=
  [``Lean.Parser.Command.definition, ``Lean.Parser.Command.theorem,
   ``Lean.Parser.Command.abbrev, ``Lean.Parser.Command.structure,
   ``Lean.Parser.Command.inductive, ``Lean.Parser.Command.classInductive]

/-- Syntax that may appear nowhere in a formalisation. Besides these, any
`do` block (a kind whose last part starts with `doSeq`), any `set_option`,
Aesop's rule clauses (`aesop (add …)`), and `decide` with a configuration
(`decide +native`) are refused. Interpolated strings (`s!"…{e}…"`, `m!`,
`f!`) are refused because the text gate in Python reads them as plain
strings, while Lean elaborates the terms inside the braces. String and
character literals are refused: a string can carry free text, and a
character literal (`!'"'`) once made the Python reader take a comment for a
string. `Name` literals (`` `foo ``, `` ``foo ``) are refused: written with
`«»` they hold any text, and the definition that holds one is printed in the
prompts. Doc comments are refused like any other comment. `open … renaming` and
`open … hiding` are refused because the statement text is cut from inside
the `open … in`, so a renaming would change what that text means. With code
generation off none of them can
run model code; the list is a second guard, and it gives the model a reason
it can act on. -/
def deniedKinds : List Name :=
  [``Lean.Parser.Term.attributes, ``Lean.Parser.Command.unsafe,
   ``Lean.Parser.Command.partial, ``Lean.Parser.Tactic.runTac,
   `Lean.Parser.Term.byElab, ``Lean.Parser.Tactic.nativeDecide,
   `Mathlib.Meta.eval_expr, `plausibleSyntax, interpolatedStrKind, strLitKind, charLitKind,
   ``Lean.Parser.Command.docComment,
   ``Lean.Parser.Command.openRenaming, ``Lean.Parser.Command.openHiding,
   ``Lean.Parser.Term.quotedName, ``Lean.Parser.Term.doubleQuotedName]

/-- The identifiers in `s` written with `«»`, with their lines' syntax. -/
partial def quotedIdents (s : Syntax) (acc : Array Syntax) : Array Syntax :=
  match s with
  | .node _ _ args => args.foldl (fun a c => quotedIdents c a) acc
  | .ident _ raw _ _ => if raw.toString.contains '«' then acc.push s else acc
  | _ => acc

partial def kinds (s : Syntax) (acc : Array (SyntaxNodeKind × Syntax)) :
    Array (SyntaxNodeKind × Syntax) :=
  match s with
  | .node _ k args => args.foldl (fun a c => kinds c a) (acc.push (k, s))
  | _ => acc

def arityOf (sig : Syntax) : Nat :=
  sig[0].getArgs.foldl (fun n b =>
    let k := b.getKind
    if k == ``Lean.Parser.Term.explicitBinder || k == ``Lean.Parser.Term.implicitBinder
        || k == ``Lean.Parser.Term.strictImplicitBinder then n + b[1].getNumArgs
    else n + 1) 0

def lastPart (k : Name) : String :=
  match k with
  | .str _ s => s
  | _ => ""

/-- Why `node`, of kind `kind`, is refused, if it is. -/
def refusal (kind : Name) (node : Syntax) : Option String :=
  let last := lastPart kind
  if last.startsWith "doSeq" then some "a `do` block"
  else if kind == interpolatedStrKind then some "an interpolated string"
  else if kind == strLitKind then some "a string literal"
  else if kind == charLitKind then some "a character literal"
  else if kind == ``Lean.Parser.Term.quotedName || kind == ``Lean.Parser.Term.doubleQuotedName then
    some "a `Name` literal"
  else if kind == ``Lean.Parser.Command.docComment then some "a comment"
  else if kind == ``Lean.Parser.Command.openRenaming then some "`open … renaming`"
  else if kind == ``Lean.Parser.Command.openHiding then some "`open … hiding`"
  else if deniedKinds.contains kind || last == "set_option" then some s!"`{kind}`"
  else if kind.getRoot == `Aesop && last.startsWith "tactic_clause" then
    some "an Aesop rule clause"
  else if kind == ``Lean.Parser.Tactic.decide
      && (kinds node #[]).any (·.1 == ``Lean.Parser.Tactic.configItem) then
    some "`decide` with a configuration"
  else none

/-- Whether `ss`, the whitespace Lean's lexer kept beside a token, holds a
comment. Only whitespace and comments go there. -/
def commentIn (ss : Substring.Raw) : Bool :=
  let t := ss.toString
  t.contains "--" || t.contains "/-"

/-- Whether Lean's lexer read a comment anywhere in `s`: in the whitespace
before or after any of its tokens. -/
partial def hasComment (s : Syntax) : Bool :=
  let inInfo : SourceInfo → Bool := fun
    | .original leading _ trailing _ => commentIn leading || commentIn trailing
    | _ => false
  match s with
  | .node info _ args => inInfo info || args.any hasComment
  | .atom info _ => inInfo info
  | .ident info _ _ _ => inInfo info
  | .missing => false

/-- The namespaces whose scoped declarations `o`, an `open` command or the
`open` of an `open … in`, makes active: those of `open A B` and
`open scoped A B`, as Lean resolves them where `o` is. An unknown namespace
is left out; the compile reports it. -/
def openedNamespaces (o : Syntax) : CommandElabM (Array Name) := do
  let decl := o[1]
  let ids : Array Syntax :=
    if decl.getKind == ``Lean.Parser.Command.openSimple then decl[0].getArgs
    else if decl.getKind == ``Lean.Parser.Command.openScoped then decl[1].getArgs
    else #[]
  let env ← getEnv
  let ns ← getCurrNamespace
  let opens ← getOpenDecls
  let mut out : Array Name := #[]
  for id in ids do
    for n in ResolveName.resolveNamespace env ns opens id.getId do
      if !out.contains n then out := out.push n
  return out

syntax (name := commandCheckCmd) "#minilab_command_check " str str : command

/-- The commands that open or close a scope. -/
def scopeCommands : List Name :=
  [``Lean.Parser.Command.open, ``Lean.Parser.Command.namespace,
   ``Lean.Parser.Command.section, ``Lean.Parser.Command.end]

/-- Whether Lean treats `n` as internal: a part starts with `_`, or the last
part is `eq_N`, `match_N`, `proof_N` or `omega_N` (`Name.isInternalDetail`).
The probe skips such names, so a formalisation may not declare them. -/
def internalName (n : Name) : Bool :=
  n.isInternal || n.isInternalDetail || n.components.any fun c => match c with
    | .str _ s => s.startsWith "_"
    | _ => false

/-- Whether the last part of `n` is `conjecture`. -/
def endsInConjecture (n : Name) : Bool :=
  match n with
  | .str _ s => s == "conjecture"
  | _ => false

/-- Parse `source` with the environment's syntax, without elaborating it
(only its `open`, `namespace`, `section` and `end` commands, so scoped
notation parses). `conjecture` counts only where its full name is
`conjecture`: inside a `section` it does; inside a `namespace`, or written
`Foo.conjecture`, it is refused. An `open` must not outlast the file: it
must be `open … in` or inside a `section` or `namespace`.
No comment may appear anywhere: Lean's lexer finds them, so a trick that
fools a reader of the text cannot hide one.
Print `PROBLEM` for each refusal, then `ARITY` (how many binders
`conjecture` has before its colon, or null), `STATEMENT` (the UTF-8 byte
range of `conjecture` from its first token to the end of its type, or
null), `OPENS` (the namespaces whose scoped declarations are active at
`conjecture`: those of its `open … in` and of the `open`s of the sections
around it), then `DONE`. The scopes the
source opened are closed again before the command ends. -/
@[command_elab commandCheckCmd] def elabCommandCheck : CommandElab := fun stx => do
  let some marker := stx[1].isStrLit? | throwError "expected the marker"
  let some src := stx[2].isStrLit? | throwError "expected the source"
  let inputCtx := Parser.mkInputContext src "<formalisation>"
  let (header, mps, msgs) ← Parser.parseHeader inputCtx
  let mut mps := mps
  let mut msgs := msgs
  let mut problems : Array String := #[]
  let saved ← getScopes
  let mut arity : Option Nat := none
  let mut statement : Option (Nat × Nat) := none
  let mut sectionOpens : Array (Nat × Name) := #[]
  let mut statementOpens : Array Name := #[]
  let mut commented := hasComment header
  let lineOf : Syntax → Nat := fun s => (inputCtx.fileMap.toPosition (s.getPos?.getD 0)).line
  repeat
    let scope ← getScope
    let pmctx : Parser.ParserModuleContext :=
      { env := ← getEnv, options := scope.opts, currNamespace := scope.currNamespace,
        openDecls := scope.openDecls }
    let (cmd, mps', msgs') := Parser.parseCommand inputCtx pmctx mps msgs
    mps := mps'
    msgs := msgs'
    if hasComment cmd then commented := true
    if Parser.isTerminalCommand cmd then break
    let line := lineOf cmd
    -- `open … in` around a command: check the `open`, then the command.
    let mut inner := cmd
    let mut opens : Array Syntax := #[]
    while inner.getKind == ``Lean.Parser.Command.in do
      opens := opens.push inner[0]
      inner := inner[2]
    let k := inner.getKind
    let mut fine := true
    for o in opens do
      if o.getKind != ``Lean.Parser.Command.open then
        problems := problems.push s!"line {line}: only `open … in` may wrap a command"
        fine := false
    if !opens.isEmpty && scopeCommands.contains k then
      problems := problems.push
        s!"line {line}: `open … in` may not wrap `open`, `namespace`, `section`, or `end`"
      fine := false
    if !allowedCommands.contains k then
      problems := problems.push s!"line {line}: the command `{k}` is not allowed"
      fine := false
    else if k == ``Lean.Parser.Command.declaration
        && !allowedDeclarations.contains inner[1].getKind then
      let dk := inner[1].getKind
      problems := problems.push s!"line {line}: the declaration kind `{dk}` is not allowed"
      fine := false
    for (kind, node) in kinds cmd #[] do
      if let some w := refusal kind node then
        let ln := if (lastPart kind).startsWith "doSeq" then line else lineOf node
        let p := s!"line {ln}: {w} is not allowed"
        if !problems.contains p then problems := problems.push p
        fine := false
      let full := (← getCurrNamespace) ++ node[0].getId
      if kind == ``Lean.Parser.Command.declId && internalName node[0].getId then
        let p := s!"line {lineOf node}: the name `{node[0].getId}` is one Lean treats as \
          internal (a part starts with `_`, or ends `eq_N`, `match_N`, `proof_N`, `omega_N`)"
        if !problems.contains p then problems := problems.push p
        fine := false
      if kind == ``Lean.Parser.Command.declId && endsInConjecture full && full != `conjecture then
        let p := s!"line {lineOf node}: `{full}`: only the theorem at the root may be named \
          `conjecture`"
        if !problems.contains p then problems := problems.push p
        fine := false
      if kind == ``Lean.Parser.Command.declId && full == `conjecture then
        let sig := (kinds cmd #[]).find? (fun p => p.1 == ``Lean.Parser.Command.declSig)
        let mut here := sectionOpens.map (·.2)
        for o in opens do
          for n in ← openedNamespaces o do
            if !here.contains n then here := here.push n
        statementOpens := here
        match sig with
        | some (_, s) =>
          arity := some (arityOf s)
          match inner.getPos?, s.getTailPos? with
          | some a, some b => statement := some (a.byteIdx, b.byteIdx)
          | _, _ => pure ()
        | none => pure ()
    -- A name written with «» can be a keyword (`«at»`) or hold any text, so
    -- it breaks the checks' Lean text and can carry text into the prompts.
    for q in quotedIdents cmd #[] do
      let p := s!"line {lineOf q}: the name `{((q.getSubstring? false false).map (·.toString)).getD ""}` \
        is written with «»; use a plain name"
      if !problems.contains p then problems := problems.push p
      fine := false
    if fine && k == ``Lean.Parser.Command.open && (← getScopes).length == saved.length then
      problems := problems.push s!"line {line}: a top-level `open` stays in force for the \
        checks after the file; use `open … in`, or put the `open` inside a `section`"
      fine := false
    -- Lean logs, not throws, an error in these commands (an unknown
    -- namespace); the compile reports it.
    if fine && scopeCommands.contains k then
      if k == ``Lean.Parser.Command.open then
        let depth := (← getScopes).length
        for n in ← openedNamespaces inner do
          sectionOpens := sectionOpens.push (depth, n)
      elabCommand cmd
      let depth := (← getScopes).length
      sectionOpens := sectionOpens.filter (·.1 ≤ depth)
  if commented then
    problems := problems.push "the file has a comment; write no comments"
  for m in msgs.toList do
    if m.severity == .error then
      problems := problems.push s!"Lean could not parse the file: {(← m.toString).take 300}"
  if (← getScopes).length != saved.length then
    problems := problems.push "every `namespace` and `section` must be closed with `end`"
  modify fun st => { st with scopes := saved }
  for p in problems.toList.take 20 do
    say marker "PROBLEM" (toJson (p.take 300))
  say marker "ARITY" (toJson arity)
  say marker "STATEMENT" (match statement with
    | some (a, b) => Json.arr #[toJson a, toJson b]
    | none => Json.null)
  say marker "OPENS" (toJson (statementOpens.map Name.toString))
  say marker "DONE" Json.null

-- ## The probe

/-- The longest binder or conclusion type printed, in characters. -/
def maxTypeChars : Nat := 1500

/-- The most file definitions one binder, the conclusion, or a definition lists. -/
def maxDefsListed : Nat := 50

/-- The longest definition printed, in characters. -/
def maxDefChars : Nat := 2000

syntax (name := probeCmd) "#minilab_probe " str num str : command

/-- Classes whose instances cannot change what a statement means, because any
two of them are equal. `open Classical` fills these with an instance that a
search after the file does not find. -/
def subsingletonClasses : List Name :=
  [``Decidable, ``DecidableEq, ``DecidablePred, ``DecidableRel]

/-- The instance arguments in `e` that Lean's instance search would not give
there: a structure written in place (`haveI : LT ℝ := ⟨…⟩`), a `let`, or an
instance of another type that unfolds to this one (`OrderDual.instLE`). The
printed type does not show instance arguments, so such an argument changes
the meaning of a statement without changing its text. `let` and `have` (a
`let` that its body does not depend on) are unfolded first, so their values
are checked where they are used. The instance the search finds may mention
a variable of a class type only if it is in `allowed`, the theorem's own
binders: such a variable bound inside the statement (`fun (i : LT ℝ) => …`,
`∃ i : LE ℝ, …`, a `match` alternative) is a local instance there, so the
search would find it and pass whatever value fills it. A variable of any
other type bound inside the statement may appear (`Fin.fintype n` under
`∀ n : ℕ`). -/
def foreignInstances (allowed : Array FVarId) (e : Expr) : MetaM (Array String) := do
  let found ← IO.mkRef (#[] : Array String)
  let pre : Expr → MetaM TransformStep := fun t => do
    if let .letE _ _ v b _ := t then return .visit (b.instantiate1 v)
    if t.isHeadBetaTarget then return .visit t.headBeta
    return .continue
  let post : Expr → MetaM TransformStep := fun t => do
    if t.isApp then
      let f := t.getAppFn
      let args := t.getAppArgs
      let info ← getFunInfoNArgs f args.size
      for i in [0:min info.paramInfo.size args.size] do
        if !info.paramInfo[i]!.binderInfo.isInstImplicit then continue
        let a := args[i]!
        let cls ← instantiateForall (← inferType f) (args.extract 0 i)
        let cls := cls.bindingDomain!
        if subsingletonClasses.any (cls.getAppFn.isConstOf ·) then continue
        let fine ← withNewMCtxDepth do
          match ← trySynthInstance cls with
          | .some s =>
            let mut inner := false
            for f in (collectFVars {} s).fvarSet.toList do
              if allowed.contains f then continue
              match ← f.findDecl? with
              | some d => if (← isClass? d.type).isSome then inner := true
              | none => inner := true
            if inner then pure false else isDefEq a s
          | _ => pure false
        if !fine then
          found.modify (·.push ((toString (← ppExpr a)).take 200).toString)
    return .done t
  let _ ← Meta.transform e (pre := pre) (post := post)
  found.get

/-- The most names `SORRY` lists. -/
def maxSorryListed : Nat := 20

/-- `e`, a type, as `(a : A) [B] … : C`: its binders with their names and
types, then the rest. -/
def ppSignature (e : Expr) : MetaM String :=
  forallTelescope e fun xs body => do
    let mut parts : Array String := #[]
    for x in xs do
      let d ← x.fvarId!.getDecl
      let n := binderName d.userName
      let ty := toString (← ppExpr d.type)
      let inner := if n == "" then ty else s!"{n} : {ty}"
      parts := parts.push (match d.binderInfo with
        | .implicit => "{" ++ inner ++ "}"
        | .strictImplicit => "⦃" ++ inner ++ "⦄"
        | .instImplicit => "[" ++ inner ++ "]"
        | .default => "(" ++ inner ++ ")")
    let head := " ".intercalate parts.toList
    return (if head == "" then "" else head ++ " ") ++ ": " ++ toString (← ppExpr body)

/-- A definition of the file as Lean prints it: `def N : T := V`, or for a
type, `inductive N : T` with a line `| C (a : A) … : T'` for each
constructor (a structure's fields are its constructor's binders). -/
def ppDefinition (ci : ConstantInfo) : MetaM String := do
  let env ← getEnv
  let user := ((privateToUserName? ci.name).getD ci.name).toString
  match ci with
  | ConstantInfo.inductInfo v =>
    let mut out := s!"inductive {user} {← ppSignature ci.type}"
    for c in v.ctors do
      if let some cc := env.find? c then
        out := out ++ s!"\n  | {((privateToUserName? c).getD c).toString} {← ppSignature cc.type}"
    return out
  | _ =>
    match ci.value? with
    | some v => return s!"def {user} : {← ppExpr ci.type} :=\n  {← ppExpr v}"
    | none => return s!"def {user} : {← ppExpr ci.type}"

/-- The foreign instances (`foreignInstances`, through `scan`) in `e`, a
definition's type or value (`isLambda`) or a constructor's type, under its
own binders. Like the theorem's binders, the definition's own arguments of a
class type are instances the search may find. -/
def foreignUnder (scan : Array FVarId → Expr → TermElabM (Array String)) (e : Expr)
    (isLambda : Bool) : TermElabM (Array String) := do
  let k : Array Expr → Expr → TermElabM (Array String) := fun xs b => do
    let ids := xs.map (·.fvarId!)
    let mut out : Array String := #[]
    for i in [0:xs.size] do
      out := out ++ (← scan (ids.extract 0 i) (← ids[i]!.getType))
    return out ++ (← scan ids b)
  if isLambda then lambdaTelescope e k else forallTelescope e k

/-- Print `SORRY`, the constants of the file other than `conjecture` that
depend on `sorryAx` (internal and generated ones too); `DEF` for each
definition the file wrote, with its text as Lean prints it; one `BINDER`
for each of the first `arity` binders of `conjecture`; and `BODY` for the
rest of its statement, with whether the statement itself depends on
`sorryAx`. Print `NONE` in place of `BINDER` and `BODY` if there is no
`conjecture`. Binder types are shown without `optParam` and `autoParam`.

Each `DEF` lists the foreign instances (`foreignInstances`) in the
definition's type and value, or in each constructor's type: Lean prints a
definition without its instance arguments, so `@LT.lt ℝ ⟨…⟩ 0 x` would
print as `0 < x`. A `BINDER` is an instance if it is written `[…]` or its
type is a class: either way it is a local instance in the statement.

`opens` are the namespaces the command check found active at `conjecture`
(`OPENS`). Their scoped instances are made active while the instance
arguments of the statement and of every definition are checked, as they
were when Lean elaborated the statement, and inactive again afterwards. A
definition elaborated under other scoped instances is refused. Types are printed outside them,
so a name an `open` brought in is printed in full (`Trick.cos`). -/
def probe (marker : String) (arity : Nat) (opens : Array Name) : CommandElabM Unit := do
  let env ← getEnv
  -- Instances are checked with the statement's scoped instances active.
  for ns in opens do
    activateScoped ns
  let active ← getEnv
  setEnv env
  let foreignIn : Array FVarId → Expr → TermElabM (Array String) := fun allowed e => do
    let printed ← getEnv
    setEnv active
    try foreignInstances allowed e finally setEnv printed
  let mut withSorry : Array String := #[]
  for (name, _) in env.constants.map₂.toList do
    if name != `conjecture && (← collectAxioms name).contains ``sorryAx then
      withSorry := withSorry.push ((privateToUserName? name).getD name).toString
  say marker "SORRY" (toJson ((withSorry.qsort (· < ·)).toList.take maxSorryListed))
  let mut fileDefs : Array Name := #[]
  for (name, ci) in env.constants.map₂.toList do
    let user := (privateToUserName? name).getD name
    if user.isInternal || user.isInternalDetail then continue
    if isAuxRecursor env name || isNoConfusion env name || isMatcherCore env name
        || (env.isProjectionFn name) || isInstanceCore env name then continue
    -- Lean records where each declaration the source wrote is; the ones it
    -- generates (`noConfusionType`, `ctorIdx`, deriving helpers) have no range.
    if (← findDeclarationRanges? name).isNone then continue
    match ci with
    | ConstantInfo.defnInfo _ | ConstantInfo.inductInfo _ => fileDefs := fileDefs.push name
    | _ => pure ()
  let userName : Name → String := fun n => ((privateToUserName? n).getD n).toString
  let mentions : Expr → Json := fun e =>
    toJson ((fileDefs.filter (fun n => e.getUsedConstants.contains n)).toList.take maxDefsListed
      |>.map userName)
  for name in fileDefs do
    let some ci := env.find? name | continue
    let ax ← collectAxioms name
    let uses := match ci with
      | ConstantInfo.inductInfo v =>
        -- A type's meaning is in its constructors.
        v.ctors.foldl (fun e c => match env.find? c with
          | some cc => mkApp e cc.type
          | none => e) ci.type
      | _ => ci.value?.getD ci.type
    let text ← liftTermElabM (ppDefinition ci)
    let foreign ← liftTermElabM do
      let mut out ← foreignUnder foreignIn ci.type false
      match ci with
      | ConstantInfo.inductInfo v =>
        for c in v.ctors do
          if let some cc := env.find? c then
            out := out ++ (← foreignUnder foreignIn cc.type false)
      | _ =>
        if let some v := ci.value? then
          out := out ++ (← foreignUnder foreignIn v true)
      pure out
    say marker "DEF" (Json.mkObj [("name", toJson (userName name)),
      ("type", toJson (ci matches ConstantInfo.inductInfo _)),
      ("sorry", toJson (ax.contains ``sorryAx)), ("uses", mentions uses),
      ("foreign", toJson foreign),
      ("text", toJson (text.take maxDefChars)),
      ("truncated", toJson (decide (text.length > maxDefChars)))])
  match env.find? `conjecture with
  | none => say marker "NONE" Json.null
  | some info =>
    let isThm := match info with
      | ConstantInfo.thmInfo _ => true
      | _ => false
    let mut typeSorry := false
    for c in info.type.getUsedConstants do
      if (← collectAxioms c).contains ``sorryAx then typeSorry := true
    liftTermElabM do
      forallBoundedTelescope info.type (some arity) fun xs body => do
        if xs.size != arity then
          throwError "`conjecture` has fewer binders than its header"
        let ids := xs.map (·.fvarId!)
        for i in [0:xs.size] do
          let d ← xs[i]!.fvarId!.getDecl
          let ty := (← instantiateMVars d.type).consumeTypeAnnotations
          let mut deps : Array Nat := #[]
          for j in [0:i] do
            if ty.containsFVar xs[j]!.fvarId! then deps := deps.push j
          let isP ← Meta.isProp ty
          let s := toString (← ppExpr ty)
          say marker "BINDER" (Json.mkObj [("index", toJson i),
              ("name", toJson (binderName d.userName)),
              ("prop", toJson isP),
              ("instance", toJson (d.binderInfo.isInstImplicit || (← isClass? ty).isSome)),
              ("deps", toJson deps), ("defs", mentions ty),
              ("foreign", toJson (← foreignIn (ids.extract 0 i) ty)),
              ("type", toJson (s.take maxTypeChars)),
              ("truncated", toJson (decide (s.length > maxTypeChars)))])
        let b ← instantiateMVars body
        let mut deps : Array Nat := #[]
        for j in [0:xs.size] do
          if b.containsFVar xs[j]!.fvarId! then deps := deps.push j
        let s := toString (← ppExpr b)
        say marker "BODY" (Json.mkObj [("theorem", toJson isThm), ("sorry", toJson typeSorry),
            ("deps", toJson deps),
            ("defs", mentions b), ("foreign", toJson (← foreignIn ids b)),
            ("type", toJson (s.take maxTypeChars)),
            ("truncated", toJson (decide (s.length > maxTypeChars)))])

/-- Run `probe`. Lean prints the types of `fun`, `∃` and other inner
binders (`pp.funBinderTypes`): `∃ y, …` hides whether `y` is an integer or
a real. It also prints the type of every numeral (`pp.numericTypes`): with
no type, `x ^ (1 / 2)` has a natural-number exponent, 0, and prints the same
as the square root `x ^ ((1 : ℝ) / 2)`; `∃ ε > 0, …` hides whether `ε` is a
natural number. -/
@[command_elab probeCmd] def elabProbe : CommandElab := fun stx => do
  let some marker := stx[1].isStrLit? | throwError "expected the marker"
  let some arity := stx[2].isNatLit? | throwError "expected the arity"
  let some opensText := stx[3].isStrLit? | throwError "expected the opens"
  let opens ← match (do (← Json.parse opensText).getArr?) with
    | .ok items => items.mapM fun j => match j.getStr? with
      | .ok t => pure t.toName
      | .error e => throwError "the opens could not be read: {e}"
    | .error e => throwError "the opens could not be read: {e}"
  let printing : Options → Options := fun o =>
    (o.setBool `pp.funBinderTypes true).setBool `pp.numericTypes true
  withScope (fun sc => { sc with opts := printing sc.opts }) (probe marker arity opens)

-- ## The vacuity goals

syntax (name := vacuityCmd) "#minilab_vacuity " str num str : command

/-- Read the witness: a JSON array of `[name, Lean text, integral]`. -/
def readWitness (text : String) : Except String (Array (String × String × Bool)) := do
  let items ← (← Json.parse text).getArr?
  items.mapM fun item => do
    let name ← (← item.getArrVal? 0).getStr?
    let value ← (← item.getArrVal? 1).getStr?
    let integral ← (← item.getArrVal? 2).getBool?
    pure (name, value, integral)

/-- Declare `MinilabContraGoal_<nonce>`, the statement `∀ binders, False`,
and for each hypothesis `i` of `conjecture` the definition
`MinilabWitness_<nonce>_<i>`: the hypothesis with the witness's values
substituted, or `True` if some variable it mentions has no value. Print
`PROP` for each hypothesis, `SUBST` for each variable a value was tried on,
and `READY` at the end. A value is elaborated against the variable's type,
and only into ℕ, ℤ, ℚ, ℝ, or `Fin k →` one of those; a fraction never goes
into ℕ or ℤ. A binder with no name takes no value. -/
@[command_elab vacuityCmd] def elabVacuity : CommandElab := fun stx => do
  let some nonce := stx[1].isStrLit? | throwError "expected the nonce"
  let some arity := stx[2].isNatLit? | throwError "expected the arity"
  let some witnessText := stx[3].isStrLit? | throwError "expected the witness"
  let witness ← match readWitness witnessText with
    | .ok w => pure w
    | .error e => throwError "the witness could not be read: {e}"
  let m := "MINILAB_VAC_" ++ nonce ++ "_"
  let env ← getEnv
  let some info := env.find? `conjecture | throwError "no conjecture"
  let scalar : Expr → Bool := fun t =>
    t.isConstOf `Real || t.isConstOf ``Rat || t.isConstOf ``Nat || t.isConstOf ``Int
  let integral : Expr → Bool := fun t => t.isConstOf ``Nat || t.isConstOf ``Int
  let define : Name → Expr → TermElabM Unit := fun name value => do
    let ps := (collectLevelParams {} value).params.toList
    let dv : DefinitionVal := { name := name, levelParams := ps, type := mkSort Level.zero,
                                value := value, hints := ReducibilityHints.abbrev,
                                safety := DefinitionSafety.safe }
    addDecl (Declaration.defnDecl dv)
  liftTermElabM do
    forallBoundedTelescope info.type (some arity) fun xs _ => do
      if xs.size != arity then throwError "`conjecture` has fewer binders than its header"
      define (Name.mkSimple s!"MinilabContraGoal_{nonce}") (← mkForallFVars xs (mkConst ``False))
      let mut fromV : Array Expr := #[]
      let mut toV : Array Expr := #[]
      for i in [0:xs.size] do
        let x := xs[i]!
        let d ← x.fvarId!.getDecl
        let raw := (← instantiateMVars d.type).consumeTypeAnnotations
        let ty := raw.replaceFVars fromV toV
        let closed := !ty.hasFVar && !ty.hasMVar
        if ← isProp raw then
          define (Name.mkSimple s!"MinilabWitness_{nonce}_{i}")
            (if closed then ty else mkConst ``True)
          say m "PROP" (Json.mkObj [("index", toJson i), ("instantiated", toJson closed)])
        else if d.binderInfo.isInstImplicit then
          if closed then
            -- Search without the binders: inside them, the search finds this
            -- binder itself, and an empty class would look filled.
            match ← withLCtx {} {} (trySynthInstance ty) with
            | LOption.some e =>
              if e.hasFVar || e.hasMVar then continue
              fromV := fromV.push x
              toV := toV.push e
              say m "SUBST" (Json.mkObj [("index", toJson i), ("ok", toJson true)])
            | _ => pure ()
        else
          let key := binderName d.userName
          match witness.find? (fun w => key != "" && w.1 == key) with
          | none => pure ()
          | some (_, text, isInt) =>
            let codomain := if ty.isForall && !ty.bindingBody!.hasLooseBVars
              && ty.bindingDomain!.isAppOfArity ``Fin 1 then ty.bindingBody! else ty
            if !(closed && scalar codomain) then
              say m "SUBST" (Json.mkObj [("index", toJson i), ("ok", toJson false),
                ("detail", toJson "its type is not ℕ, ℤ, ℚ, ℝ, or a vector of one of them")])
            else if !isInt && integral codomain then
              say m "SUBST" (Json.mkObj [("index", toJson i), ("ok", toJson false),
                ("detail", toJson "the value is a fraction and the variable's type is ℕ or ℤ")])
            else
              match Parser.runParserCategory (← getEnv) `term text with
              | Except.error e =>
                say m "SUBST" (Json.mkObj [("index", toJson i), ("ok", toJson false),
                  ("detail", toJson e)])
              | Except.ok vstx =>
                try
                  let e ← Term.elabTermEnsuringType vstx ty
                  Term.synthesizeSyntheticMVarsNoPostponing
                  let e ← instantiateMVars e
                  if e.hasMVar || e.hasFVar then throwError "the value did not elaborate fully"
                  fromV := fromV.push x
                  toV := toV.push e
                  say m "SUBST" (Json.mkObj [("index", toJson i), ("ok", toJson true)])
                catch ex =>
                  say m "SUBST" (Json.mkObj [("index", toJson i), ("ok", toJson false),
                    ("detail", toJson ((← ex.toMessageData.toString).take 300))])
      say m "READY" Json.null

end Minilab.Guard
