/* Smart Product Pricing — frontend logic */

const EXAMPLES = [
  {
    name: "Taco Sauce 6-pack",
    actual: 4.89,
    text: "Item Name: La Victoria Green Taco Sauce Mild, 12 Ounce (Pack of 6)\nValue: 72.0\nUnit: Fl Oz\n",
    url: "https://m.media-amazon.com/images/I/51mo8htwTHL.jpg",
  },
  {
    name: "Chef-size Basil",
    actual: 18.5,
    text: "Item Name: Member's Mark Member's Mark, Basil, 6.25 oz\nBullet Point 1: Green Herb, Italian Staple, Great mixed with Oregano\nBullet Point 2: Large Size, Chef Bottle\nBullet Point 3: Packed in the USA\nValue: 6.25\nUnit: ounce\n",
    url: "https://m.media-amazon.com/images/I/81nw0HXpCRL.jpg",
  },
  {
    name: "Cooking Wine case",
    actual: 66.49,
    text: "Item Name: kedem Sherry Cooking Wine, 12.7 Ounce - 12 per case.\nBullet Point: kedem Sherry Cooking Wine, 12.7 Ounce - 12 per case.\nValue: 12.0\nUnit: Count\n",
    url: "https://m.media-amazon.com/images/I/41sA037+QvL.jpg",
  },
  {
    name: "Cider Vinegar 102oz",
    actual: 81.44,
    text: "Item Name: Organic Vinegar; Apple Cider\nValue: 102.0\nUnit: Fl Oz\n",
    url: "https://m.media-amazon.com/images/I/41SHfxsFz5L.jpg",
  },
];

const $ = (id) => document.getElementById(id);

const textInput = $("text-input");
const imageInput = $("image-input");
const predictBtn = $("predict-btn");
const thumb = $("thumb");
const thumbImg = $("thumb-img");

const states = {
  idle: $("tag-idle"),
  loading: $("tag-loading"),
  result: $("tag-result"),
  error: $("tag-error"),
};

let currentExample = null; // holds actual price when an example is loaded

function showState(name) {
  Object.entries(states).forEach(([key, el]) => (el.hidden = key !== name));
}

function updateThumb() {
  const url = imageInput.value.trim();
  if (url && url.startsWith("http")) {
    thumbImg.src = url;
    thumb.hidden = false;
  } else {
    thumb.hidden = true;
  }
}

imageInput.addEventListener("input", () => {
  currentExample = null;
  updateThumb();
});
textInput.addEventListener("input", () => (currentExample = null));
thumbImg.addEventListener("error", () => (thumb.hidden = true));

// build example chips
const chips = $("example-chips");
EXAMPLES.forEach((ex) => {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "chip";
  btn.innerHTML = `${ex.name} · <b>$${ex.actual.toFixed(2)}</b>`;
  btn.addEventListener("click", () => {
    textInput.value = ex.text;
    imageInput.value = ex.url;
    currentExample = ex;
    updateThumb();
    predict();
  });
  chips.appendChild(btn);
});

async function predict() {
  const text = textInput.value.trim();
  if (!text) {
    $("tag-error-text").textContent = "Paste a product listing first";
    showState("error");
    return;
  }

  const example = currentExample;
  predictBtn.disabled = true;
  showState("loading");

  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, image_url: imageInput.value.trim() || null }),
    });

    if (!resp.ok) {
      const detail = (await resp.json().catch(() => ({}))).detail;
      throw new Error(detail || `Server error (${resp.status})`);
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
    $("tag-error-text").textContent = err.message || "Prediction failed";
    showState("error");
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
textInput.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") predict();
});
