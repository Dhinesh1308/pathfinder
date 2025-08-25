import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, BookOpen, GraduationCap, Home as HomeIcon, FileText, MessageSquare, Search, Plus, LogIn } from "lucide-react";

// =============================
// Tailwind note: Canvas preview includes Tailwind.
// This is a single-file demo app that simulates the full Pathfinder AI flow
// with a local, in-browser RAG index (TF-IDF based) and mock data.
// No external backend is required to run this preview.
// =============================

// ---------- Utility: Tiny RAG engine (TF-IDF over chunks) ----------
function tokenize(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

function chunkText(text, chunkSize = 600, overlap = 80) {
  // Simple char-based chunking to mimic RAG splitters
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    const end = Math.min(text.length, i + chunkSize);
    const slice = text.slice(i, end);
    chunks.push(slice.trim());
    i = end - overlap;
    if (i < 0) i = 0;
  }
  return chunks.filter(Boolean);
}

function buildTfIdfIndex(documents) {
  // documents: [{id, title, text}] -> chunk into [{id, docId, text, tokens}]
  const chunks = [];
  let cid = 0;
  for (const d of documents) {
    const parts = chunkText(d.text);
    for (const p of parts) {
      const tokens = tokenize(p);
      if (tokens.length === 0) continue;
      chunks.push({ id: `c${cid++}`, docId: d.id, title: d.title, text: p, tokens });
    }
  }
  // DF
  const df = new Map();
  for (const c of chunks) {
    const uniq = new Set(c.tokens);
    for (const t of uniq) df.set(t, (df.get(t) || 0) + 1);
  }
  const N = chunks.length || 1;
  // Precompute tf-idf vector length for each chunk
  const chunkVec = chunks.map((c) => {
    const tf = new Map();
    for (const t of c.tokens) tf.set(t, (tf.get(t) || 0) + 1);
    const vec = new Map();
    let len2 = 0;
    tf.forEach((f, t) => {
      const idf = Math.log((N + 1) / ((df.get(t) || 0) + 1)) + 1; // smooth
      const w = (f / c.tokens.length) * idf;
      vec.set(t, w);
      len2 += w * w;
    });
    return { ...c, vec, norm: Math.sqrt(len2) || 1 };
  });
  return { chunks: chunkVec, df, N };
}

function tfidfQuery(index, query, k = 4) {
  if (!index || index.chunks.length === 0) return [];
  const qTokens = tokenize(query);
  if (qTokens.length === 0) return [];
  const qtf = new Map();
  qTokens.forEach((t) => qtf.set(t, (qtf.get(t) || 0) + 1));
  const qvec = new Map();
  let qlen2 = 0;
  const { df, N } = index;
  qtf.forEach((f, t) => {
    const idf = Math.log((N + 1) / ((df.get(t) || 0) + 1)) + 1;
    const w = (f / qTokens.length) * idf;
    qvec.set(t, w);
    qlen2 += w * w;
  });
  const qnorm = Math.sqrt(qlen2) || 1;
  const scores = index.chunks.map((c) => {
    let dot = 0;
    qvec.forEach((qw, t) => {
      const cw = c.vec.get(t) || 0;
      dot += qw * cw;
    });
    const cos = dot / (qnorm * c.norm);
    return { ...c, score: cos };
  });
  return scores.sort((a, b) => b.score - a.score).slice(0, k);
}

// ---------- Storage Helpers ----------
const LS_KEY_DOCS = "pathfinder_docs_v1";
const LS_KEY_USER = "pathfinder_user_v1";

function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });
  useEffect(() => {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
  }, [key, value]);
  return [value, setValue];
}

// ---------- Sample Data ----------
const SAMPLE_EXAMS = [
  { id: "jee", name: "JEE Main", level: "UG", subject: "Engineering", month: "April", mode: "MCQ", difficulty: "Hard" },
  { id: "neet", name: "NEET UG", level: "UG", subject: "Medical", month: "May", mode: "MCQ", difficulty: "Hard" },
  { id: "gate", name: "GATE", level: "PG", subject: "Engineering", month: "Feb", mode: "MCQ/NAT", difficulty: "Hard" },
  { id: "cat", name: "CAT", level: "PG", subject: "Management", month: "Nov", mode: "MCQ/VARC", difficulty: "Medium" },
];

const SAMPLE_DOCS = [
  {
    id: "d1",
    title: "DSA Notes - Arrays & Strings",
    text:
      "Arrays store elements contiguously. Common operations: traversal, insertion, deletion, rotation. Strings are arrays of chars. Practice problems include two-sum, kadane, sliding window.",
    meta: { type: "note", tags: ["dsa", "arrays", "strings"] },
  },
  {
    id: "d2",
    title: "Operating Systems - Scheduling",
    text:
      "CPU scheduling algorithms: FCFS, SJF, SRTF, RR, Priority. Metrics: waiting time, turnaround time, response time. Gantt charts help visualize CPU allocation.",
    meta: { type: "note", tags: ["os", "scheduling"] },
  },
  {
    id: "d3",
    title: "Career: Data Scientist Pathway",
    text:
      "Foundations: Python, statistics, linear algebra. Tools: pandas, numpy, scikit-learn. Projects: EDA, ML models, dashboards. Roadmap: internships, Kaggle, portfolio, networking.",
    meta: { type: "career", tags: ["career", "ml"] },
  },
];

// ---------- PDF Parsing (best effort) ----------
// We attempt to use pdfjs if available in the preview environment. If import fails, we gracefully
// fall back to a simple message and store filename only.
async function extractPdfText(file) {
  try {
    const pdfjsLib = await import("pdfjs-dist");
    const workerSrc = await import("pdfjs-dist/build/pdf.worker.mjs");
    pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;
    const bytes = await file.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: bytes });
    const pdf = await loadingTask.promise;
    let fullText = "";
    for (let p = 1; p <= pdf.numPages; p++) {
      const page = await pdf.getPage(p);
      const content = await page.getTextContent();
      const strings = content.items.map((it) => it.str);
      fullText += strings.join(" ") + "\n";
    }
    return fullText;
  } catch (e) {
    console.warn("PDF parse failed; storing name only", e);
    return ""; // fall back
  }
}

// ---------- Components ----------
function Badge({ children }) {
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-gray-100 border border-gray-200">
      {children}
    </span>
  );
}

function Card({ title, children, footer, onClick }) {
  return (
    <motion.div
      whileHover={{ y: -2 }}
      className="rounded-2xl p-4 bg-white shadow-sm border border-gray-100 cursor-default"
      onClick={onClick}
    >
      {title && <h3 className="text-lg font-semibold mb-2">{title}</h3>}
      <div className="text-sm text-gray-700">{children}</div>
      {footer && <div className="pt-3 mt-3 border-t text-xs text-gray-500">{footer}</div>}
    </motion.div>
  );
}

function Navbar({ current, setCurrent }) {
  const tabs = [
    { id: "home", label: "Home", Icon: HomeIcon },
    { id: "exams", label: "Exams", Icon: BookOpen },
    { id: "documents", label: "Documents", Icon: FileText },
    { id: "career", label: "Career Guidance", Icon: GraduationCap },
    { id: "chat", label: "AI Chatbot", Icon: MessageSquare },
  ];
  return (
    <div className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-between h-14">
          <div className="font-bold text-xl">Pathfinder AI</div>
          <div className="flex gap-1">
            {tabs.map((t) => (
              <button
                key={t.id}
                onClick={() => setCurrent(t.id)}
                className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm ${
                  current === t.id ? "bg-gray-900 text-white" : "hover:bg-gray-100"
                }`}
              >
                <t.Icon size={16} /> {t.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Onboarding({ open, onClose, setUser }) {
  const [form, setForm] = useState({ name: "", degree: "", interests: "" });
  if (!open) return null;
  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center p-4">
      <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="bg-white rounded-2xl w-full max-w-lg p-6 shadow-xl">
        <div className="flex items-center gap-2 mb-4">
          <LogIn size={18} />
          <h2 className="text-lg font-semibold">Welcome to Pathfinder AI</h2>
        </div>
        <p className="text-sm text-gray-600 mb-4">Tell us a bit about you to personalize recommendations.</p>
        <div className="grid gap-3">
          <input className="border rounded-xl px-3 py-2" placeholder="Full name" value={form.name} onChange={(e)=>setForm({...form, name:e.target.value})} />
          <input className="border rounded-xl px-3 py-2" placeholder="Degree / Year (e.g., B.E CSE, 2nd year)" value={form.degree} onChange={(e)=>setForm({...form, degree:e.target.value})} />
          <input className="border rounded-xl px-3 py-2" placeholder="Interests (comma-separated)" value={form.interests} onChange={(e)=>setForm({...form, interests:e.target.value})} />
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <button className="px-3 py-2 rounded-xl hover:bg-gray-100" onClick={onClose}>Skip</button>
          <button
            className="px-3 py-2 rounded-xl bg-gray-900 text-white"
            onClick={() => {
              setUser({ ...form, createdAt: Date.now() });
              onClose();
            }}
          >
            Continue
          </button>
        </div>
      </motion.div>
    </div>
  );
}

function Home({ user, docs }) {
  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Hey{user?.name ? `, ${user.name}` : ""}! 👋</h1>
        <p className="text-gray-600">Your student-focused AI hub for exams, resources, and career guidance.</p>
      </div>
      <div className="grid md:grid-cols-3 gap-4">
        <Card title="Upload Study Resources" footer={<span className="inline-flex items-center gap-2"><Upload size={14}/>PDF/Notes supported</span>}>
          Add notes & PDFs to build your personal knowledge base. Our local RAG engine makes them searchable.
        </Card>
        <Card title="Exams Planner" footer="Filter by level, subject, month">
          Quickly browse exams, view past questions, and create a prep plan.
        </Card>
        <Card title="AI Chatbot (RAG)" footer="Personalized, context-aware answers">
          Ask questions—responses cite your uploaded content when relevant.
        </Card>
      </div>
      <div className="grid md:grid-cols-2 gap-4 mt-4">
        <Card title="Recent Documents">
          <ul className="list-disc pl-4">
            {docs.slice(-5).map((d)=> (
              <li key={d.id}>{d.title}</li>
            ))}
            {docs.length===0 && <li>No documents yet. Upload on the Documents page.</li>}
          </ul>
        </Card>
        <Card title="Quick Tips">
          <ul className="list-disc pl-4">
            <li>Use the Documents page to index your notes.</li>
            <li>Ask specific questions in Chat for best RAG retrieval.</li>
            <li>Filter Exams by level to focus your prep.</li>
          </ul>
        </Card>
      </div>
    </div>
  );
}

function Documents({ docs, setDocs, setIndex }) {
  const fileInput = useRef(null);
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    const s = search.trim().toLowerCase();
    if (!s) return docs;
    return docs.filter((d) => d.title.toLowerCase().includes(s) || (d.meta?.tags || []).join(",").includes(s));
  }, [docs, search]);

  async function handleFiles(files) {
    const newDocs = [];
    for (const f of files) {
      let text = "";
      if (f.type === "text/plain" || f.name.endsWith(".md")) {
        text = await f.text();
      } else if (f.type === "application/pdf" || f.name.endsWith(".pdf")) {
        text = await extractPdfText(f);
      } else {
        // try to read as text anyway
        try { text = await f.text(); } catch { text = ""; }
      }
      newDocs.push({ id: `u_${Date.now()}_${Math.random().toString(36).slice(2)}`, title: f.name, text, meta: { type: "upload", tags: [] } });
    }
    const merged = [...docs, ...newDocs];
    setDocs(merged);
    setIndex(buildTfIdfIndex(merged));
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <div className="flex items-center gap-2 mb-4">
        <div className="relative flex-1">
          <input
            className="w-full border rounded-xl pl-9 pr-3 py-2"
            placeholder="Search documents by title or tag"
            value={search}
            onChange={(e)=>setSearch(e.target.value)}
          />
          <Search size={16} className="absolute left-3 top-2.5 text-gray-400" />
        </div>
        <button className="flex items-center gap-2 px-3 py-2 rounded-xl bg-gray-900 text-white" onClick={()=>fileInput.current?.click()}>
          <Upload size={16}/> Upload
        </button>
        <input type="file" ref={fileInput} className="hidden" multiple onChange={(e)=> e.target.files && handleFiles(e.target.files)} />
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        {filtered.map((d)=>(
          <Card key={d.id} title={d.title} footer={<div className="flex gap-2 flex-wrap">{(d.meta?.tags||[]).map((t)=>(<Badge key={t}>{t}</Badge>))}</div>}>
            <div className="line-clamp-3">{d.text ? d.text.slice(0, 180) + (d.text.length>180?"…":"") : <em className="text-gray-500">(No extracted text. PDF images or encrypted PDFs may not parse in preview.)</em>}</div>
          </Card>
        ))}
        {filtered.length===0 && (
          <div className="text-gray-500 text-sm">No documents match your search.</div>
        )}
      </div>
    </div>
  );
}

function Exams() {
  const [query, setQuery] = useState("");
  const [level, setLevel] = useState("All");

  const items = useMemo(()=>{
    return SAMPLE_EXAMS.filter((e)=>{
      const okLevel = level === "All" || e.level === level;
      const okQuery = !query || e.name.toLowerCase().includes(query.toLowerCase()) || e.subject.toLowerCase().includes(query.toLowerCase());
      return okLevel && okQuery;
    });
  }, [query, level]);

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <div className="flex flex-wrap items-center gap-2 mb-4">
        <input className="border rounded-xl px-3 py-2" placeholder="Search exams" value={query} onChange={(e)=>setQuery(e.target.value)} />
        <select className="border rounded-xl px-3 py-2" value={level} onChange={(e)=>setLevel(e.target.value)}>
          <option>All</option>
          <option>UG</option>
          <option>PG</option>
        </select>
        <Badge>Past Qs available</Badge>
      </div>
      <div className="grid md:grid-cols-2 gap-4">
        {items.map((e)=>(
          <Card key={e.id} title={e.name} footer={<div className="flex gap-2 flex-wrap"><Badge>{e.level}</Badge><Badge>{e.subject}</Badge><Badge>{e.month}</Badge><Badge>{e.mode}</Badge><Badge>{e.difficulty}</Badge></div>}>
            <div className="space-y-2">
              <p>Overview: {e.name} is a {e.level} level exam for {e.subject} aspirants typically held in {e.month}.</p>
              <div className="flex gap-2">
                <button className="px-3 py-2 rounded-xl bg-gray-900 text-white text-sm">View Syllabus</button>
                <button className="px-3 py-2 rounded-xl border text-sm">Past Questions</button>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

function Career() {
  const tracks = [
    { id: "ds", title: "Data Scientist", points: ["Python & Stats", "EDA & ML", "Projects & Portfolio", "Networking"], tags: ["ML", "Pandas" ] },
    { id: "se", title: "Software Engineer", points: ["DSA", "System Design Basics", "Git & CI/CD", "Web/Backend Projects"], tags: ["DSA", "Web"] },
    { id: "cy", title: "Cybersecurity Analyst", points: ["Networks", "Linux & Scripting", "Vuln Assessment", "CTFs"], tags: ["Security", "Linux"] },
  ];
  const [filter, setFilter] = useState("");
  const shown = tracks.filter(t => t.title.toLowerCase().includes(filter.toLowerCase()) || t.tags.join(",").toLowerCase().includes(filter.toLowerCase()))
  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <div className="mb-4 flex items-center gap-2">
        <input className="border rounded-xl px-3 py-2" placeholder="Filter (e.g., ML, Web)" value={filter} onChange={(e)=>setFilter(e.target.value)} />
      </div>
      <div className="grid md:grid-cols-3 gap-4">
        {shown.map((t)=>(
          <Card key={t.id} title={t.title} footer={<div className="flex gap-2 flex-wrap">{t.tags.map(x=> <Badge key={x}>{x}</Badge>)}</div>}>
            <ul className="list-disc pl-4">
              {t.points.map((p,i)=>(<li key={i}>{p}</li>))}
            </ul>
          </Card>
        ))}
      </div>
    </div>
  );
}

function Chat({ index, docs, user }) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hi! I'm your Pathfinder AI assistant. Ask me about exams, study topics, or careers. I'll cite your documents when helpful." },
  ]);

  function generateAnswer(question) {
    // RAG retrieve
    const hits = tfidfQuery(index, question, 4);

    // Safety (very light illustrative filter)
    const unsafe = /self-harm|weapon|cheat|leak/i.test(question);
    if (unsafe) {
      return {
        text: "I can't assist with that request. I can help with study plans, exam prep, or career guidance instead.",
        citations: [],
      };
    }

    if (!hits || hits.length === 0) {
      return {
        text: "Here's a general guideline based on common study strategies: break topics into chunks, practice past questions, and review mistakes. Upload notes to get context-aware answers!",
        citations: [],
      };
    }

    const snippets = hits.map((h, i) => `(${i + 1}) ${h.text.slice(0, 180)}${h.text.length > 180 ? "…" : ""}`);
    const cites = hits.map((h, i) => ({ idx: i + 1, title: h.title }));

    const answer = `Based on your materials, consider these points:\n- ${snippets.join("\n- ")}\n\nIn summary: focus on the most relevant concepts retrieved above. Ask a follow-up for details.`;
    return { text: answer, citations: cites };
  }

  return (
    <div className="max-w-3xl mx-auto px-4 py-6">
      <div className="rounded-2xl border bg-white">
        <div className="p-4 border-b flex items-center gap-2">
          <MessageSquare size={18} />
          <div className="font-semibold">AI Chatbot {user?.name ? `for ${user.name}` : "(RAG demo)"}</div>
        </div>
        <div className="p-4 space-y-4 max-h-[50vh] overflow-auto">
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm ${m.role === "user" ? "bg-gray-900 text-white" : "bg-gray-100"}`}>
                <div className="whitespace-pre-wrap">{m.text}</div>
                {m.citations && m.citations.length > 0 && (
                  <div className="text-xs text-gray-500 mt-2">
                    Sources: {m.citations.map((c) => (<span key={c.idx} className="mr-2">[{c.idx}] {c.title}</span>))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
        <div className="p-3 border-t flex items-center gap-2">
          <input
            className="flex-1 border rounded-xl px-3 py-2"
            placeholder="Ask about arrays vs strings, OS scheduling, or your career path…"
            value={input}
            onChange={(e)=>setInput(e.target.value)}
            onKeyDown={(e)=>{ if (e.key === 'Enter') { 
              const q = input.trim(); if (!q) return; setMessages((m)=>[...m, {role:'user', text:q}]);
              const { text, citations } = generateAnswer(q);
              setMessages((m)=>[...m, {role:'user', text:q}, {role:'assistant', text, citations}]);
              setInput("");
            }}}
          />
          <button
            className="px-3 py-2 rounded-xl bg-gray-900 text-white flex items-center gap-2"
            onClick={()=>{
              const q = input.trim(); if (!q) return; setMessages((m)=>[...m, {role:'user', text:q}]);
              const { text, citations } = generateAnswer(q);
              setMessages((m)=>[...m, {role:'user', text:q}, {role:'assistant', text, citations}]);
              setInput("");
            }}
          >
            <MessageSquare size={16}/> Ask
          </button>
        </div>
      </div>
      <div className="text-xs text-gray-500 mt-2">Note: This canvas preview runs a local RAG (TF‑IDF). In production, connect FastAPI + LangChain + FAISS/Chroma + GPT.</div>
    </div>
  );
}

export default function App() {
  const [current, setCurrent] = useState("home");
  const [user, setUser] = useLocalStorage(LS_KEY_USER, null);
  const [docs, setDocs] = useLocalStorage(LS_KEY_DOCS, SAMPLE_DOCS);
  const [index, setIndex] = useState(null);
  const [showOnboard, setShowOnboard] = useState(!user);

  useEffect(() => { setIndex(buildTfIdfIndex(docs)); }, []); // initial from SAMPLE_DOCS
  useEffect(() => { setIndex(buildTfIdfIndex(docs)); }, [docs]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <Navbar current={current} setCurrent={setCurrent} />
      <Onboarding open={showOnboard} onClose={() => setShowOnboard(false)} setUser={setUser} />

      <AnimatePresence mode="wait">
        {current === "home" && (
          <motion.div key="home" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
            <Home user={user} docs={docs} />
          </motion.div>
        )}
        {current === "exams" && (
          <motion.div key="exams" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
            <Exams />
          </motion.div>
        )}
        {current === "documents" && (
          <motion.div key="docs" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
            <Documents docs={docs} setDocs={setDocs} setIndex={setIndex} />
          </motion.div>
        )}
        {current === "career" && (
          <motion.div key="career" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
            <Career />
          </motion.div>
        )}
        {current === "chat" && (
          <motion.div key="chat" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
            <Chat index={index} docs={docs} user={user} />
          </motion.div>
        )}
      </AnimatePresence>

      <footer className="max-w-6xl mx-auto px-4 py-10 text-xs text-gray-500">
        <div className="border-t pt-6">© {new Date().getFullYear()} Pathfinder AI • Next.js + FastAPI in prod • Canvas demo uses React + Tailwind + Framer Motion</div>
      </footer>
    </div>
  );
}
