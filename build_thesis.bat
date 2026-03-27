@echo off
echo ========================================================
echo GRAPH RAG THESIS BUILD SCRIPT
echo ========================================================
echo.
echo Yeu cau: Ban can cai dat MiKTeX hoac TeX Live tren Windows.
echo.

cd rag2025

echo Dang bien dich Luan an (Lan 1/2)...
pdflatex -interaction=nonstopmode thesis_latex.tex

echo Dang bien dich Luan an (Lan 2/2) de cap nhat Muc luc va Reference...
pdflatex -interaction=nonstopmode thesis_latex.tex

echo Dang bien dich Slide Bao ve (Lan 1)...
pdflatex -interaction=nonstopmode thesis_defense_slides.tex

echo Dang bien dich Slide Bao ve (Lan 2)...
pdflatex -interaction=nonstopmode thesis_defense_slides.tex

echo.
echo Dọn dẹp các file rác (aux, log, toc, out)...
del *.aux *.log *.toc *.out *.snm *.nav 2>nul

echo ========================================================
echo DONE! File thesis_latex.pdf va thesis_defense_slides.pdf da san sang!
echo Neu he thong bao loi pdflatex not found, hay upload cac file .tex len Overleaf.com de bien dich online.
echo ========================================================
pause