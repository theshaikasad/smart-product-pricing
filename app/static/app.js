/* Smart Product Pricing — frontend logic */

const EXAMPLES = [
  {
    name: "Taco Sauce 6-pack",
    actual: 4.89,
    item: "La Victoria Green Taco Sauce Mild, 12 Ounce (Pack of 6)",
    bullets: [],
    description: "",
    value: "72.0",
    unit: "Fl Oz",
    url: "https://m.media-amazon.com/images/I/51mo8htwTHL.jpg",
  },
  {
    name: "Chef-size Basil",
    actual: 18.5,
    item: "Member's Mark Member's Mark, Basil, 6.25 oz",
    bullets: [
      "Green Herb, Italian Staple, Great mixed with Oregano",
      "Large Size, Chef Bottle",
      "Packed in the USA",
    ],
    description: "",
    value: "6.25",
    unit: "ounce",
    url: "https://m.media-amazon.com/images/I/81nw0HXpCRL.jpg",
  },
  {
    name: "Cooking Wine case",
    actual: 66.49,
    item: "kedem Sherry Cooking Wine, 12.7 Ounce - 12 per case.",
    bullets: ["kedem Sherry Cooking Wine, 12.7 Ounce - 12 per case."],
    description: "",
    value: "12.0",
    unit: "Count",
    url: "https://m.media-amazon.com/images/I/41sA037+QvL.jpg",
  },
  {
    name: "Cider Vinegar 102oz",
    actual: 81.44,
    item: "Organic Vinegar; Apple Cider",
    bullets: [],
    description: "",
    value: "102.0",
    unit: "Fl Oz",
    url: "https://m.media-amazon.com/images/I/41SHfxsFz5L.jpg",
  },
];

const $ = (id) => document.getElementById(id);

const tabForm = $("tab-form");
const tabRaw = $("tab-raw");
const modeForm = $("mode-form");
const modeRaw = $("mode-raw");

const itemName = $("item-name");
const bullets = $("bullets");
const description = $("description");
const value = $("value");
const unit = $("unit");
const textInput = $("text-input");

const imageInput = $("image-input");
const imageFile = $("image-file");
const uploadNote = $("upload-note");
const predictBtn = $("predict-btn");
const thumb = $("thumb");
const thumbImg = $("thumb-img");

const states = {
  idle: $("tag-idle"),
  loading: $("tag-loading"),
  result: $("tag-result"),
  error: $("tag-error"),
};

let rawMode = false;
let currentExample = null; // holds actual price when an example is loaded
let uploadedImage = null; // { b64, name }

function showState(name) {
  Object.entries(states).forEach(([key, el]) => (el.hidden = key !== name));
}

function setMode(raw) {
  rawMode = raw;
  tabForm.classList.toggle("active", !raw);
  tabRaw.classList.toggle("active", raw);
  tabForm.setAttribute("aria-selected", String(!raw));
  tabRaw.setAttribute("aria-selected", String(raw));
  modeForm.hidden = raw;
  modeRaw.hidden = !raw;
  // carry the form over to raw text so switching never loses work
  if (raw && !textInput.value.trim()) textInput.value = composeFromFields();
}

tabForm.addEventListener("click", () => setMode(false));
tabRaw.addEventListener("click", () => setMode(true));

/* Build catalog_content in the dataset's own format */
function composeFromFields() {
  const lines = [];
  if (itemName.value.trim()) lines.push(`Item Name: ${itemName.value.trim()}`);

  const bl = bullets.value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (bl.length === 1) {
    lines.push(`Bullet Point: ${bl[0]}`);
  } else {
    bl.forEach((b, i) => lines.push(`Bullet Point ${i + 1}: ${b}`));
  }

  if (description.value.trim()) lines.push(`Product Description: ${description.value.trim()}`);
  if (value.value.trim()) lines.push(`Value: ${value.value.trim()}`);
  if (unit.value.trim()) lines.push(`Unit: ${unit.value.trim()}`);
  return lines.length ? lines.join("\n") + "\n" : "";
}

function getCatalogText() {
  return rawMode ? textInput.value.trim() : composeFromFields().trim();
}

/* ---------- image: URL or upload ---------- */

function updateThumb() {
  if (uploadedImage) {
    thumbImg.src = "data:;base64," + uploadedImage.b64;
    thumb.hidden = false;
    return;
  }
  const url = imageInput.value.trim();
  if (url && url.startsWith("http")) {
    thumbImg.src = url;
    thumb.hidden = false;
  } else {
    thumb.hidden = true;
  }
}

function clearUpload() {
  uploadedImage = null;
  imageFile.value = "";
  uploadNote.hidden = true;
  uploadNote.textContent = "";
  updateThumb();
}

imageFile.addEventListener("change", () => {
  const file = imageFile.files[0];
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) {
    alertError("Image must be under 8 MB");
    imageFile.value = "";
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    uploadedImage = { b64: reader.result.split(",", 2)[1], name: file.name };
    currentExample = null;
    imageInput.value = "";
    uploadNote.innerHTML = `📎 ${escapeHtml(file.name)} attached<button type="button" id="clear-upload">remove</button>`;
    uploadNote.hidden = false;
    $("clear-upload").addEventListener("click", clearUpload);
    updateThumb();
  };
  reader.readAsDataURL(file);
});

imageInput.addEventListener("input", () => {
  currentExample = null;
  if (imageInput.value.trim()) clearUpload();
  updateThumb();
});

[itemName, bullets, description, value, unit, textInput].forEach((el) =>
  el.addEventListener("input", () => (currentExample = null))
);
thumbImg.addEventListener("error", () => (thumb.hidden = true));

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------- examples ---------- */

const chips = $("example-chips");
EXAMPLES.forEach((ex) => {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "chip";
  btn.innerHTML = `${ex.name} · <b>$${ex.actual.toFixed(2)}</b>`;
  btn.addEventListener("click", () => {
    itemName.value = ex.item;
    bullets.value = ex.bullets.join("\n");
    description.value = ex.description;
    value.value = ex.value;
    unit.value = ex.unit;
    setMode(false);
    textInput.value = composeFromFields();
    clearUpload();
    imageInput.value = ex.url;
    currentExample = ex;
    updateThumb();
    predict();
  });
  chips.appendChild(btn);
});

/* ---------- prediction ---------- */

function alertError(msg) {
  $("tag-error-text").textContent = msg;
  showState("error");
}

async function predict() {
  const text = getCatalogText();
  if (!text) {
    alertError(rawMode ? "Paste a product listing first" : "Give the product a name first");
    return;
  }

  const example = currentExample;
  predictBtn.disabled = true;
  showState("loading");

  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        image_url: uploadedImage ? null : imageInput.value.trim() || null,
        image_b64: uploadedImage ? uploadedImage.b64 : null,
      }),
    });

    if (!resp.ok) {
      const detail = (await resp.json().catch(() => ({}))).detail;
      throw new Error(typeof detail === "string" ? detail : `Server error (${resp.status})`);
    }

    const data = await resp.json();

    $("tag-price").textContent = formatPrice(data.price);
    $("meta-log").textContent = data.log_price.toFixed(3);
    $("meta-image").textContent = data.image_used ? "used ✓" : "not used";
    $("meta-latency").textContent = `${Math.round(data.latency_ms)} ms`;

    const actualEl = $("tag-actual");
    if (example) {
      const err = Math.abs(data.price - example.actual) / example.actual * 100;
      actualEl.innerHTML = `dataset price: <b>$${example.actual.toFixed(2)}</b> · off by ${err.toFixed(1)}%`;
      actualEl.hidden = false;
    } else {
      actualEl.hidden = true;
    }

    // restart the stamp animation
    const priceEl = $("tag-price");
    priceEl.style.animation = "none";
    void priceEl.offsetWidth;
    priceEl.style.animation = "";

    showState("result");
  } catch (err) {
    alertError(err.message || "Prediction failed");
  } finally {
    predictBtn.disabled = false;
  }
}

function formatPrice(p) {
  const opts = p >= 1000
    ? { maximumFractionDigits: 0 }
    : { minimumFractionDigits: 2, maximumFractionDigits: 2 };
  return "$" + p.toLocaleString("en-US", opts);
}

predictBtn.addEventListener("click", predict);
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") predict();
});
