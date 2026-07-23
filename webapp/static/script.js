const form = document.getElementById("loanForm");
const submitBtn = document.getElementById("submitBtn");
const formError = document.getElementById("formError");

const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultState = document.getElementById("resultState");

const stampEl = document.getElementById("stampEl");
const probValue = document.getElementById("probValue");
const riskValue = document.getElementById("riskValue");
const gaugeFill = document.getElementById("gaugeFill");
const resultNote = document.getElementById("resultNote");

const modelDot = document.getElementById("modelDot");
const modelName = document.getElementById("modelName");
const fileNo = document.getElementById("fileNo");

// Random-ish file number, just cosmetic (ledger flavor)
fileNo.textContent = "LD-" + Math.floor(100000 + Math.random() * 900000);

// ---- Check backend / model health on load ----
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.model_loaded) {
      modelDot.classList.add("ok");
      modelName.textContent = `desk ready — ${data.model_name}`;
    } else {
      modelDot.classList.add("err");
      modelName.textContent = "model not trained yet";
    }
  } catch (e) {
    modelDot.classList.add("err");
    modelName.textContent = "backend unreachable";
  }
}
checkHealth();

const notes = {
  Low: "Strong profile. Low likelihood of default under current terms.",
  Moderate: "Acceptable risk. Consider standard terms or minor rate adjustment.",
  High: "Elevated risk. Recommend manual review, collateral, or reduced amount.",
  "Very High": "Significant risk signal. Recommend decline or substantial mitigation.",
};

function resetPanels() {
  emptyState.classList.add("hidden");
  loadingState.classList.remove("hidden");
  resultState.classList.add("hidden");
  stampEl.classList.remove("show", "approved", "declined");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  formError.textContent = "";
  submitBtn.disabled = true;
  submitBtn.textContent = "Reviewing…";
  resetPanels();

  const payload = {
    person_age: Number(document.getElementById("person_age").value),
    person_income: Number(document.getElementById("person_income").value),
    person_home_ownership: document.getElementById("person_home_ownership").value,
    person_emp_length: Number(document.getElementById("person_emp_length").value),
    loan_intent: document.getElementById("loan_intent").value,
    loan_grade: document.getElementById("loan_grade").value,
    loan_amnt: Number(document.getElementById("loan_amnt").value),
    loan_int_rate: Number(document.getElementById("loan_int_rate").value),
    loan_percent_income: Number(document.getElementById("loan_percent_income").value),
    cb_person_default_on_file: document.querySelector('input[name="default_on_file"]:checked').value,
    cb_person_cred_hist_length: Number(document.getElementById("cb_person_cred_hist_length").value),
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    renderResult(data);
  } catch (err) {
    loadingState.classList.add("hidden");
    emptyState.classList.remove("hidden");
    formError.textContent = err.message || "Something went wrong. Please try again.";
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Send file for review";
  }
});

function renderResult(data) {
  loadingState.classList.add("hidden");
  resultState.classList.remove("hidden");

  const approved = data.verdict === "APPROVED";
  stampEl.textContent = data.verdict;
  stampEl.classList.add(approved ? "approved" : "declined");

  // trigger stamp animation on next frame
  requestAnimationFrame(() => stampEl.classList.add("show"));

  const pct = (data.default_probability * 100).toFixed(1);
  probValue.textContent = `${pct}%`;
  riskValue.textContent = data.risk_level;
  gaugeFill.style.width = `${Math.min(data.default_probability * 100, 100)}%`;
  resultNote.textContent = notes[data.risk_level] || "";
}
