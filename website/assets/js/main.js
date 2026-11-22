(() => {
  // 1. Interactive feature showcase tabs
  const tabs = Array.from(document.querySelectorAll(".showcase-tab"));
  const panels = Array.from(document.querySelectorAll(".showcase-panel"));

  function activateTab(tab) {
    const targetId = tab.getAttribute("aria-controls");
    tabs.forEach((item) => {
      const selected = item === tab;
      item.setAttribute("aria-selected", selected ? "true" : "false");
      item.tabIndex = selected ? 0 : -1;
    });
    panels.forEach((panel) => {
      panel.hidden = panel.id !== targetId;
    });
  }

  tabs.forEach((tab, idx) => {
    tab.addEventListener("click", () => activateTab(tab));
    tab.addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        const delta = event.key === "ArrowRight" ? 1 : -1;
        const next = tabs[(idx + delta + tabs.length) % tabs.length];
        activateTab(next);
        next.focus();
      }
    });
  });

  // 2. Click-to-copy buttons for commands and BibTeX blocks
  document.querySelectorAll("[data-copy-target]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const selector = btn.getAttribute("data-copy-target");
      const target = selector ? document.querySelector(selector) : null;
      if (!target) return;
      const text = target.textContent ? target.textContent.trim() : "";
      if (!text) return;
      const original = btn.textContent;
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = "Copied";
      } catch (_err) {
        btn.textContent = "Copy failed";
      }
      window.setTimeout(() => {
        btn.textContent = original;
      }, 1600);
    });
  });

  // 3. Click-to-zoom lightbox for showcase screenshots and simulation previews
  const lightbox = document.getElementById("annolidLightbox");
  const lightboxImg = document.getElementById("annolidLightboxImg");
  const lightboxCaption = document.getElementById("annolidLightboxCaption");
  const lightboxClose = document.getElementById("annolidLightboxClose");

  function closeLightbox() {
    if (!lightbox) return;
    lightbox.hidden = true;
    document.body.style.overflow = "";
  }

  function openLightbox(src, alt) {
    if (!lightbox || !lightboxImg) return;
    lightboxImg.src = src;
    lightboxImg.alt = alt || "";
    if (lightboxCaption) {
      lightboxCaption.textContent = alt || "";
    }
    lightbox.hidden = false;
    document.body.style.overflow = "hidden";
  }

  document.querySelectorAll("img[data-zoomable]").forEach((img) => {
    img.addEventListener("click", () => {
      openLightbox(img.currentSrc || img.src, img.alt);
    });
  });

  if (lightboxClose) {
    lightboxClose.addEventListener("click", closeLightbox);
  }
  if (lightbox) {
    lightbox.addEventListener("click", (event) => {
      if (event.target === lightbox) {
        closeLightbox();
      }
    });
  }
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && lightbox && !lightbox.hidden) {
      closeLightbox();
    }
  });
})();
