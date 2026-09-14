#!/usr/bin/env bash
# Produce a marked-up copy of the manuscript showing every change against the
# submitted version: additions in blue, deletions in red.
#
# The editor asks for revisions to be highlighted, and it is the only practical
# way to review a revision that touches prose, equations and tables at once.
# The output is a separate document, so the manuscript itself stays clean.
#
#   bash routines/mark_changes.sh [baseline-git-ref]
#
# The default baseline is the commit holding the version that was submitted.

set -eu

MANUSCRIPT="${MANUSCRIPT:-/d/bin/HMM}"
MAIN="${MAIN:-ManuscriptEJOR}"
BASELINE="${1:-6983044}"

# MiKTeX ships latexdiff twice: the plain script needs Algorithm::Diff from CPAN,
# which is generally not installed, while the -so build embeds it.
LATEXDIFF="/c/Users/franc/AppData/Local/Programs/MiKTeX/miktex/bin/x64/latexdiff-so.exe"

cd "$MANUSCRIPT"

git show "${BASELINE}:${MAIN}.tex" > "/tmp/${MAIN}_baseline.tex"

# --append-safecmd stops latexdiff marking up the inside of a \revnote. The note
# explains a change; it should read as a note rather than as inserted text.
"$LATEXDIFF" --encoding=utf8 --type=CFONT --math-markup=whole \
    --append-safecmd="revnote" \
    "/tmp/${MAIN}_baseline.tex" "${MAIN}.tex" > "${MAIN}_changes.tex" 2>/dev/null

# Three passes plus bibtex: the marked-up document has its own aux and
# bibliography, so citations resolve only after the bibliography is built.
#
# pdflatex returns a non-zero status for unresolved references, which is the
# normal state of affairs on the earlier passes, so its exit code is not a useful
# signal here. Whether the run succeeded is judged from the output instead.
for pass in 1 2 3; do
    # Defining \revnotesON turns on the margin annotations, which stay inert when
    # the manuscript itself is compiled.
    pdflatex -interaction=nonstopmode -jobname="${MAIN}_changes" \
        "\\def\\revnotesON{}\\input{${MAIN}_changes.tex}" > /dev/null 2>&1 || true
    if [ "$pass" = "1" ]; then
        bibtex "${MAIN}_changes" > /dev/null 2>&1 || true
    fi
done

# Checking that the file merely exists is not enough: if a viewer holds the PDF
# open, pdflatex cannot write to it and silently leaves the previous version in
# place, which is the one way this script could report success while handing back
# stale output. The log of the final pass is the reliable signal.
if ! grep -q "Output written on ${MAIN}_changes.pdf" "${MAIN}_changes.log" 2>/dev/null; then
    echo "the marked-up PDF was not written." >&2
    grep -m2 -E "^!|I can't write on file" "${MAIN}_changes.log" 2>/dev/null >&2 || true
    echo "if it is open in a viewer, close it and run this again." >&2
    exit 1
fi

added=$(grep -c 'DIFaddbegin' "${MAIN}_changes.tex" || true)
deleted=$(grep -c 'DIFdelbegin' "${MAIN}_changes.tex" || true)

echo "baseline: ${BASELINE}"
echo "marked-up document: ${MANUSCRIPT}/${MAIN}_changes.pdf"
echo "  ${added} insertions, ${deleted} deletions"
