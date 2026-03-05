const elements = document.querySelectorAll(".reveal");

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("on");
      }
    });
  },
  { threshold: 0.14 }
);

elements.forEach((el, i) => {
  el.style.transitionDelay = `${80 * i}ms`;
  observer.observe(el);
});
