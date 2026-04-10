import { laws } from "./data/laws.js";

const state = {
  activeCategory: "全部",
  activeLawId: laws[0]?.id ?? null,
};

const elements = {
  filterBar: document.querySelector("#filter-bar"),
  lawGrid: document.querySelector("#law-grid"),
  spotlight: document.querySelector("#spotlight"),
  starterCount: document.querySelector("#starter-count"),
  regionCount: document.querySelector("#region-count"),
  categoryCount: document.querySelector("#category-count"),
  reviewCount: document.querySelector("#review-count"),
  year: document.querySelector("#year"),
};

const categories = ["全部", ...new Set(laws.map((law) => law.category))];

function getVisibleLaws() {
  if (state.activeCategory === "全部") {
    return laws;
  }

  return laws.filter((law) => law.category === state.activeCategory);
}

function renderStats() {
  const regions = new Set(laws.map((law) => law.region));
  const reviewCount = laws.filter((law) => law.verification !== "已补官方来源").length;

  elements.starterCount.textContent = String(laws.length);
  elements.regionCount.textContent = String(regions.size);
  elements.categoryCount.textContent = String(categories.length - 1);
  elements.reviewCount.textContent = String(reviewCount);
}

function renderFilters() {
  elements.filterBar.innerHTML = categories
    .map((category) => {
      const activeClass = category === state.activeCategory ? " is-active" : "";

      return `
        <button
          type="button"
          class="filter-chip${activeClass}"
          data-category="${category}"
          aria-pressed="${category === state.activeCategory}"
        >
          ${category}
        </button>
      `;
    })
    .join("");

  elements.filterBar.querySelectorAll("[data-category]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeCategory = button.dataset.category;
      const visibleLaws = getVisibleLaws();
      state.activeLawId = visibleLaws[0]?.id ?? null;
      renderFilters();
      renderLaws();
      renderSpotlight();
    });
  });
}

function renderLaws() {
  const visibleLaws = getVisibleLaws();

  elements.lawGrid.innerHTML = visibleLaws
    .map((law) => {
      const activeClass = law.id === state.activeLawId ? " is-active" : "";
      const dangerClass = law.riskLevel === "高" ? " is-danger" : "";

      return `
        <button type="button" class="law-card${activeClass}" data-law-id="${law.id}">
          <div class="law-topline">
            <span class="law-region">${law.region}</span>
            <span class="pill${dangerClass}">风险 ${law.riskLevel}</span>
          </div>
          <h3>${law.title}</h3>
          <p>${law.summary}</p>
          <div class="law-meta">
            <span class="meta-tag">${law.category}</span>
            <span class="meta-tag">${law.verification}</span>
            <span class="meta-tag">${law.sourceHint}</span>
          </div>
        </button>
      `;
    })
    .join("");

  elements.lawGrid.querySelectorAll("[data-law-id]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeLawId = button.dataset.lawId;
      renderLaws();
      renderSpotlight();
    });
  });
}

function renderSpotlight() {
  const visibleLaws = getVisibleLaws();
  const activeLaw = visibleLaws.find((law) => law.id === state.activeLawId) ?? visibleLaws[0];

  if (!activeLaw) {
    elements.spotlight.innerHTML = `
      <p class="panel-label">暂无内容</p>
      <p class="spotlight-summary">当前筛选条件下没有可显示的条目。</p>
    `;
    return;
  }

  state.activeLawId = activeLaw.id;

  elements.spotlight.innerHTML = `
    <div class="spotlight-topline">
      <span class="pill">${activeLaw.category}</span>
      <span class="pill${activeLaw.riskLevel === "高" ? " is-danger" : ""}">风险 ${activeLaw.riskLevel}</span>
    </div>
    <p class="panel-label">${activeLaw.region}</p>
    <h3>${activeLaw.title}</h3>
    <p class="spotlight-summary">${activeLaw.summary}</p>

    <section class="spotlight-section">
      <p class="spotlight-label">为什么容易踩雷</p>
      <p class="spotlight-detail">${activeLaw.risk}</p>
    </section>

    <section class="spotlight-section">
      <p class="spotlight-label">建议继续核验</p>
      <p class="spotlight-detail">${activeLaw.verification}</p>
    </section>

    <section class="spotlight-section">
      <p class="spotlight-label">展示字段示例</p>
      <p class="spotlight-detail">潜在后果：${activeLaw.penalty}</p>
      <p class="spotlight-detail">旅行提示：${activeLaw.travelerTip}</p>
      <p class="spotlight-detail">来源方向：${activeLaw.sourceHint}</p>
    </section>
  `;
}

function setupReveal() {
  const revealTargets = document.querySelectorAll(".reveal");

  if (!("IntersectionObserver" in window)) {
    revealTargets.forEach((target) => target.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) {
          return;
        }

        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.16 }
  );

  revealTargets.forEach((target) => observer.observe(target));
}

function init() {
  renderStats();
  renderFilters();
  renderLaws();
  renderSpotlight();
  setupReveal();
  elements.year.textContent = String(new Date().getFullYear());
}

init();
