(function () {
  const button = document.getElementById("themeToggle");
  if (!button) return;
  const saved = localStorage.getItem("stocker-theme");
  if (saved === "dark") document.documentElement.dataset.theme = "dark";
  if (saved === "light") document.documentElement.dataset.theme = "light";
  button.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme !== "dark";
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("stocker-theme", dark ? "dark" : "light");
  });
})();
