const sections = document.querySelectorAll('.reveal');

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in');
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.14 }
);

sections.forEach((section, index) => {
  section.style.transitionDelay = `${index * 70}ms`;
  observer.observe(section);
});
