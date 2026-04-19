# latexmk 4.87 can run bibtex before pdflatex when report.aux is absent: it still reads a
# stale report.log from an earlier build and mis-orders rules (bibtex sees a stub .aux).
# If there is no .aux yet, drop the old log (and any leftover .bbl) so the first step is pdflatex.
BEGIN {
  if (! -f 'report.aux') {
    unlink 'report.log' if -f 'report.log';
    unlink 'report.bbl' if -f 'report.bbl';
  }
}
