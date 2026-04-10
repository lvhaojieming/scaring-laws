import { useDeferredValue, useState } from "react";
import { LAW_ENTRIES, REGIONS, STATUSES } from "./data/laws";
import type { LawEntry, ReviewStatus } from "./types";

const STATUS_TEXT: Record<ReviewStatus, string> = {
  verified: "已核验",
  "needs-source": "待补来源",
  contested: "存在争议",
};

const STATUS_CLASS: Record<ReviewStatus, string> = {
  verified: "badge badge-verified",
  "needs-source": "badge badge-needs-source",
  contested: "badge badge-contested",
};

function App() {
  const [query, setQuery] = useState("");
  const [selectedRegion, setSelectedRegion] =
    useState<(typeof REGIONS)[number]>("全部");
  const [selectedStatus, setSelectedStatus] =
    useState<(typeof STATUSES)[number]["value"]>("all");
  const deferredQuery = useDeferredValue(query.trim().toLowerCase());

  const filteredEntries = LAW_ENTRIES.filter((entry) => {
    const matchesQuery =
      deferredQuery.length === 0 ||
      [entry.title, entry.country, entry.summary, entry.whyItFeelsScary]
        .join(" ")
        .toLowerCase()
        .includes(deferredQuery);

    const matchesRegion =
      selectedRegion === "全部" || entry.region === selectedRegion;

    const matchesStatus =
      selectedStatus === "all" || entry.reviewStatus === selectedStatus;

    return matchesQuery && matchesRegion && matchesStatus;
  });

  const featuredEntry = filteredEntries[0] ?? LAW_ENTRIES[0];
  const verifiedCount = LAW_ENTRIES.filter(
    (entry) => entry.reviewStatus === "verified",
  ).length;
  const highRiskCount = LAW_ENTRIES.filter((entry) => entry.severity >= 8).length;
  const regionCount = new Set(LAW_ENTRIES.map((entry) => entry.region)).size;

  return (
    <div className="page-shell">
      <div className="ambient ambient-left" aria-hidden="true" />
      <div className="ambient ambient-right" aria-hidden="true" />

      <header className="topbar">
        <div>
          <p className="eyebrow">Risk Research Interface</p>
          <h1>Scaring Laws</h1>
        </div>
        <div className="topbar-note">
          一个可继续扩展成法规案例库、内容审核台或旅行风险地图的起始版本
        </div>
      </header>

      <main className="content">
        <section className="hero">
          <div className="hero-copy">
            <p className="hero-kicker">把“可怕但说不清”的法律风险，整理成一个能查能筛的项目。</p>
            <h2>从法律惊悚感，切到产品化的研究流程。</h2>
            <p className="hero-text">
              这一版先把项目底盘搭起来：主题首页、示例数据、筛选检索和核验状态。
              你后面可以很自然地接数据库、CMS、地图、AI 摘要或者来源管理。
            </p>

            <div className="search-panel">
              <label className="control search-control">
                <span>检索关键词</span>
                <input
                  type="search"
                  placeholder="例如：言论、阿联酋、平台审核"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </label>

              <div className="filter-row">
                <label className="control">
                  <span>地区</span>
                  <select
                    value={selectedRegion}
                    onChange={(event) =>
                      setSelectedRegion(event.target.value as (typeof REGIONS)[number])
                    }
                  >
                    {REGIONS.map((region) => (
                      <option key={region} value={region}>
                        {region}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="control">
                  <span>核验状态</span>
                  <select
                    value={selectedStatus}
                    onChange={(event) =>
                      setSelectedStatus(
                        event.target.value as (typeof STATUSES)[number]["value"],
                      )
                    }
                  >
                    {STATUSES.map((status) => (
                      <option key={status.value} value={status.value}>
                        {status.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </div>
          </div>

          <aside className="hero-feature">
            <div className="hero-feature-header">
              <p>本次焦点案例</p>
              <span className={STATUS_CLASS[featuredEntry.reviewStatus]}>
                {STATUS_TEXT[featuredEntry.reviewStatus]}
              </span>
            </div>
            <h3>{featuredEntry.title}</h3>
            <p className="hero-feature-meta">
              {featuredEntry.country} / {featuredEntry.category} / 风险指数{" "}
              {featuredEntry.severity}/10
            </p>
            <p>{featuredEntry.summary}</p>
            <div className="signal-box">
              <strong>为什么让人发怵</strong>
              <p>{featuredEntry.whyItFeelsScary}</p>
            </div>
            <div className="signal-box signal-box-muted">
              <strong>下一步研究</strong>
              <p>{featuredEntry.researchLead}</p>
            </div>
          </aside>
        </section>

        <section className="stats-grid" aria-label="项目概览">
          <article className="stat-card">
            <span>法律线索</span>
            <strong>{LAW_ENTRIES.length}</strong>
            <p>当前用于原型演示的数据条目数量</p>
          </article>
          <article className="stat-card">
            <span>高风险案例</span>
            <strong>{highRiskCount}</strong>
            <p>风险指数大于等于 8 的线索</p>
          </article>
          <article className="stat-card">
            <span>已核验条目</span>
            <strong>{verifiedCount}</strong>
            <p>已经标记为可继续扩展来源页的内容</p>
          </article>
          <article className="stat-card">
            <span>覆盖地区</span>
            <strong>{regionCount}</strong>
            <p>可继续接入更多国家与地区的法规样本</p>
          </article>
        </section>

        <section className="section-heading">
          <div>
            <p className="eyebrow">Research Queue</p>
            <h2>案例列表</h2>
          </div>
          <p>
            演示数据默认混合了已核验、待补来源和存在争议三种状态，方便你后面继续接人工研究流程。
          </p>
        </section>

        <section className="card-grid">
          {filteredEntries.length > 0 ? (
            filteredEntries.map((entry) => <LawCard key={entry.id} entry={entry} />)
          ) : (
            <div className="empty-state">
              <p>没有匹配到结果。</p>
              <span>试试换个地区，或者把关键词缩短一点。</span>
            </div>
          )}
        </section>

        <section className="workflow">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Next Expansion</p>
              <h2>下一步最自然的扩展方向</h2>
            </div>
          </div>
          <div className="workflow-grid">
            <article>
              <span>01</span>
              <h3>来源页</h3>
              <p>给每条法规补官方法条、新闻报道、判例摘要和更新时间。</p>
            </article>
            <article>
              <span>02</span>
              <h3>后台录入</h3>
              <p>把示例数据换成数据库或 Headless CMS，方便持续扩展内容。</p>
            </article>
            <article>
              <span>03</span>
              <h3>风险评分</h3>
              <p>继续拆成执行强度、处罚力度、适用模糊度和跨境影响四个维度。</p>
            </article>
          </div>
        </section>
      </main>
    </div>
  );
}

function LawCard({ entry }: { entry: LawEntry }) {
  return (
    <article className="law-card">
      <div className="law-card-header">
        <div>
          <p className="law-card-country">{entry.country}</p>
          <h3>{entry.title}</h3>
        </div>
        <span className={STATUS_CLASS[entry.reviewStatus]}>
          {STATUS_TEXT[entry.reviewStatus]}
        </span>
      </div>

      <div className="meta-row">
        <span>{entry.region}</span>
        <span>{entry.category}</span>
        <span>{entry.year}</span>
      </div>

      <p className="law-summary">{entry.summary}</p>

      <div className="risk-meter" aria-label={`风险指数 ${entry.severity} / 10`}>
        <div className="risk-meter-track">
          <div
            className="risk-meter-fill"
            style={{ width: `${entry.severity * 10}%` }}
          />
        </div>
        <strong>{entry.severity}/10</strong>
      </div>

      <div className="detail-block">
        <strong>可能影响</strong>
        <p>{entry.impact}</p>
      </div>
      <div className="detail-block detail-block-alt">
        <strong>研究提示</strong>
        <p>{entry.researchLead}</p>
      </div>
    </article>
  );
}

export default App;

