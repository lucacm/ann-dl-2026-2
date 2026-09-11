window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // Padrão oficial do Material para pymdownx.arithmatex (generic: true): só
    // tipografa dentro de <span class="arithmatex">, ignora o resto da página.
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Re-typeset after each page load (Material SPA)
document$.subscribe(() => {
  MathJax.typesetPromise();
});
