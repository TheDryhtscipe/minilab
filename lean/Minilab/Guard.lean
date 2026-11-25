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
- `#minilab_probe "marker" arity`: the binders and conclusion of
  `conjecture`, and the file's definitions.
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
strings, while Lean elaborates the terms inside the braces. With code generation off none of them can
run model code; the list is a second guard, and it gives the model a reason
it can act on. -/
def deniedKinds : List Name :=
  [``Lean.Parser.Term.attributes, ``Lean.Parser.Command.unsafe,
   ``Lean.Parser.Command.partial, ``Lean.Parser.Tactic.runTac,
   `Lean.Parser.Term.byElab, ``Lean.Parser.Tactic.nativeDecide,
   `Mathlib.Meta.eval_expr, `plausibleSyntax, interpolatedStrKind]

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
  else if deniedKinds.contains kind || last == "set_option" then some s!"`{kind}`"
  else if kind.getRoot == `Aesop && last.startsWith "tactic_clause" then
    some "an Aesop rule clause"
  else if kind == ``Lean.Parser.Tactic.decide
      && (kinds node #[]).any (·.1 == ``Lean.Parser.Tactic.configItem) then
    some "`decide` with a configuration"
  else none

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
Print `PROBLEM` for each refusal, then `ARITY` (how many binders
`conjecture` has before its colon, or null), `STATEMENT` (the UTF-8 byte
range of `conjecture` from its first token to the end of its type, or
null), then `DONE`. The scopes the
source opened are closed again before the command ends. -/
@[command_elab commandCheckCmd] def elabCommandCheck : CommandElab := fun stx => do
  let some marker := stx[1].isStrLit? | throwError "expected the marker"
  let some src := stx[2].isStrLit? | throwError "expected the source"
  let inputCtx := Parser.mkInputContext src "<formalisation>"
  let (_, mps, msgs) ← Parser.parseHeader inputCtx
  let mut mps := mps
  let mut msgs := msgs
  let mut problems : Array String := #[]
  let saved ← getScopes
  let mut arity : Option Nat := none
  let mut statement : Option (Nat × Nat) := none
  let lineOf : Syntax → Nat := fun s => (inputCtx.fileMap.toPosition (s.getPos?.getD 0)).line
  repeat
    let scope ← getScope
    let pmctx : Parser.ParserModuleContext :=
      { env := ← getEnv, options := scope.opts, currNamespace := scope.currNamespace,
        openDecls := scope.openDecls }
    let (cmd, mps', msgs') := Parser.parseCommand inputCtx pmctx mps msgs
    mps := mps'
    msgs := msgs'
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
        match sig with
        | some (_, s) =>
          arity := some (arityOf s)
          match inner.getPos?, s.getTailPos? with
          | some a, some b => statement := some (a.byteIdx, b.byteIdx)
          | _, _ => pure ()
        | none => pure ()
    if fine && k == ``Lean.Parser.Command.open && (← getScopes).length == saved.length then
      problems := problems.push s!"line {line}: a top-level `open` stays in force for the \
        checks after the file; use `open … in`, or put the `open` inside a `section`"
      fine := false
    -- Lean logs, not throws, an error in these commands (an unknown
    -- namespace); the compile reports it.
    if fine && scopeCommands.contains k then
      elabCommand cmd
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
  say marker "DONE" Json.null

-- ## The probe

/-- The longest binder or conclusion type printed, in characters. -/
def maxTypeChars : Nat := 1500

/-- The most file definitions one binder, the conclusion, or a definition lists. -/
def maxDefsListed : Nat := 50

syntax (name := probeCmd) "#minilab_probe " str num : command

/-- The most names `SORRY` lists. -/
def maxSorryListed : Nat := 20

/-- Print `SORRY`, the constants of the file other than `conjecture` that
depend on `sorryAx` (internal and generated ones too); `DEF` for each
definition the file wrote; one `BINDER` for each of the first `arity`
binders of `conjecture`; and `BODY` for the rest of its statement, with
whether the statement itself depends on `sorryAx`. Print `NONE` in place of
`BINDER` and `BODY` if there is no `conjecture`. Binder types are shown
without `optParam` and `autoParam`. -/
@[command_elab probeCmd] def elabProbe : CommandElab := fun stx => do
  let some marker := stx[1].isStrLit? | throwError "expected the marker"
  let some arity := stx[2].isNatLit? | throwError "expected the arity"
  let env ← getEnv
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
    | ConstantInfo.defnInfo _ => fileDefs := fileDefs.push name
    | _ => pure ()
  let userName : Name → String := fun n => ((privateToUserName? n).getD n).toString
  let mentions : Expr → Json := fun e =>
    toJson ((fileDefs.filter (fun n => e.getUsedConstants.contains n)).toList.take maxDefsListed
      |>.map userName)
  for name in fileDefs do
    let some ci := env.find? name | continue
    let ax ← collectAxioms name
    say marker "DEF" (Json.mkObj [("name", toJson (userName name)),
      ("sorry", toJson (ax.contains ``sorryAx)), ("uses", mentions (ci.value?.getD ci.type))])
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
              ("prop", toJson isP), ("instance", toJson d.binderInfo.isInstImplicit),
              ("deps", toJson deps), ("defs", mentions ty),
              ("type", toJson (s.take maxTypeChars)),
              ("truncated", toJson (decide (s.length > maxTypeChars)))])
        let b ← instantiateMVars body
        let mut deps : Array Nat := #[]
        for j in [0:xs.size] do
          if b.containsFVar xs[j]!.fvarId! then deps := deps.push j
        let s := toString (← ppExpr b)
        say marker "BODY" (Json.mkObj [("theorem", toJson isThm), ("sorry", toJson typeSorry),
            ("deps", toJson deps),
            ("defs", mentions b), ("type", toJson (s.take maxTypeChars)),
            ("truncated", toJson (decide (s.length > maxTypeChars)))])

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
            match ← trySynthInstance ty with
            | LOption.some e =>
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
