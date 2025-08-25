import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Upload,
  BookOpen,
  GraduationCap,
  Home as HomeIcon,
  FileText,
  MessageSquare,
  LogIn,
  Search,
} from "lucide-react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
} from "react-router-dom";

// ---------- TF-IDF Engine ----------
function tokenize(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}
function chunkText(text, chunkSize = 600, overlap = 80) {
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
  const df = new Map();
  for (const c of chunks) {
    const uniq = new Set(c.tokens);
    for (const t of uniq) df.set(t, (df.get(t) || 0) + 1);
  }
  const N = chunks.length || 1;
  const chunkVec = chunks.map((c) => {
    const tf = new Map();
    for (const t of c.tokens) tf.set(t, (tf.get(t) || 0) + 1);
    const vec = new Map();
    let len2 = 0;
    tf.forEach((f, t) => {
      const idf = Math.log((N + 1) / ((df.get(t) || 0) + 1)) + 1;
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

// ---------- Storage ----------
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
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {}
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
  { id: "d1", title: "DSA Notes - Arrays & Strings", text: "Arrays store elements contiguously..." },
  { id: "d2", title: "Operating Systems - Scheduling", text: "CPU scheduling algorithms: FCFS, SJF..." },
  { id: "d3", title: "Career: Data Scientist Pathway", text: "Foundations: Python, statistics, linear algebra..." },
];

// ---------- Components ----------
function Badge({ children }) {
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-gray-100 border">
      {children}
    </span>
  );
}
function Card({ title, children, footer }) {
  return (
    <motion.div whileHover={{ y: -2 }} className="rounded-2xl p-4 bg-white shadow-sm border">
      {title && <h3 className="text-lg font-semibold mb-2">{title}</h3>}
      <div className="text-sm text-gray-700">{children}</div>
      {footer && <div className="pt-3 mt-3 border-t text-xs text-gray-500">{footer}</div>}
    </motion.div>
  );
}
function Navbar() {
  const tabs = [
    { id: "home", label: "Home", path: "/" },
    { id: "exams", label: "Exams", path: "/exams" },
    { id: "documents", label: "Documents", path: "/documents" },
    { id: "career", label: "Career Guidance", path: "/career" },
    { id: "chat", label: "AI Chatbot", path: "/chat" },
  ];
  return (
    <div className="sticky top-0 bg-white border-b">
      <div className="flex gap-2 p-3">
        {tabs.map((t) => (
          <Link key={t.id} to={t.path} className="px-3 py-2 rounded hover:bg-gray-100">
            {t.label}
          </Link>
        ))}
      </div>
    </div>
  );
}
function Onboarding({ open, onClose, setUser }) {
  const [form, setForm] = useState({ name: "", degree: "", interests: "" });
  if (!open) return null;
  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center p-4">
      <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="bg-white rounded-2xl p-6">
        <h2 className="text-lg font-semibold flex items-center gap-2"><LogIn size={18}/> Welcome to Pathfinder AI</h2>
        <p className="text-sm text-gray-600 mb-4">Tell us a bit about you to personalize recommendations.</p>
        <div className="grid gap-3">
          <input className="border rounded px-3 py-2" placeholder="Full name" value={form.name} onChange={(e)=>setForm({...form, name:e.target.value})}/>
          <input className="border rounded px-3 py-2" placeholder="Degree / Year" value={form.degree} onChange={(e)=>setForm({...form, degree:e.target.value})}/>
          <input className="border rounded px-3 py-2" placeholder="Interests" value={form.interests} onChange={(e)=>setForm({...form, interests:e.target.value})}/>
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <button onClick={onClose} className="px-3 py-2 rounded hover:bg-gray-100">Skip</button>
          <button onClick={()=>{setUser({...form, createdAt: Date.now()}); onClose();}} className="px-3 py-2 rounded bg-gray-900 text-white">Continue</button>
        </div>
      </motion.div>
    </div>
  );
}

// ---------- Pages ----------
function Home({ user, docs }) {
  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <h1 className="text-2xl font-bold">Hey{user?.name ? `, ${user.name}` : ""}! 👋</h1>
      <p className="text-gray-600 mb-6">Your student-focused AI hub for exams, resources, and career guidance.</p>
      <div className="grid md:grid-cols-3 gap-4">
        <Card title="Upload Study Resources" footer="PDF/Notes supported">Add notes & PDFs to make them searchable.</Card>
        <Card title="Exams Planner" footer="Filter by level, subject, month">Browse exams and prep plans.</Card>
        <Card title="AI Chatbot" footer="Context-aware answers">Ask questions with citations.</Card>
      </div>
      <div className="grid md:grid-cols-2 gap-4 mt-4">
        <Card title="Recent Documents">
          <ul className="list-disc pl-4">{docs.slice(-5).map((d)=><li key={d.id}>{d.title}</li>)}</ul>
        </Card>
        <Card title="Quick Tips">
          <ul className="list-disc pl-4">
            <li>Upload notes in Documents page.</li>
            <li>Ask specific questions in Chat.</li>
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
    const s = search.toLowerCase();
    return docs.filter((d) => d.title.toLowerCase().includes(s));
  }, [docs, search]);
  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <div className="flex gap-2 mb-4">
        <input className="border rounded px-3 py-2 flex-1" placeholder="Search documents" value={search} onChange={(e)=>setSearch(e.target.value)}/>
        <button onClick={()=>fileInput.current?.click()} className="px-3 py-2 bg-gray-900 text-white rounded">Upload</button>
        <input type="file" ref={fileInput} className="hidden" multiple />
      </div>
      <div className="grid md:grid-cols-2 gap-4">
        {filtered.map((d)=><Card key={d.id} title={d.title}>{d.text.slice(0,100)}...</Card>)}
      </div>
    </div>
  );
}
function Exams() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <h1 className="text-xl font-bold mb-4">Exams</h1>
      <div className="grid md:grid-cols-2 gap-4">
        {SAMPLE_EXAMS.map((e)=>
          <Card key={e.id} title={e.name} footer={<div className="flex gap-2"><Badge>{e.level}</Badge><Badge>{e.subject}</Badge></div>}>
            Overview: {e.name} is a {e.level} level exam for {e.subject}.
          </Card>
        )}
      </div>
    </div>
  );
}
function Career() {
  return <div className="p-6">Career guidance section (customize here)</div>;
}
function Chat({ index, docs, user }) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([{ role: "assistant", text: "Hi! I'm your Pathfinder AI assistant." }]);
  function ask() {
    const q = input.trim();
    if (!q) return;
    setMessages((m)=>[...m,{role:"user",text:q}]);
    const hits = tfidfQuery(index, q, 4);
    const answer = hits.length
      ? `Based on your docs: ${hits.map((h)=>h.title).join(", ")}`
      : "General tip: break study into chunks, practice past questions.";
    setMessages((m)=>[...m,{role:"user",text:q},{role:"assistant",text:answer}]);
    setInput("");
  }
  return (
    <div className="max-w-3xl mx-auto px-4 py-6">
      <div className="border rounded p-4 bg-white">
        {messages.map((m,i)=><div key={i} className={m.role==="user"?"text-right":""}><div className="my-2">{m.text}</div></div>)}
        <div className="flex gap-2 mt-4">
          <input className="border rounded px-3 py-2 flex-1" value={input} onChange={(e)=>setInput(e.target.value)} onKeyDown={(e)=>e.key==="Enter"&&ask()}/>
          <button onClick={ask} className="px-3 py-2 bg-gray-900 text-white rounded">Ask</button>
        </div>
      </div>
    </div>
  );
}

// ---------- App ----------
export default function App() {
  const [user, setUser] = useLocalStorage(LS_KEY_USER, null);
  const [docs, setDocs] = useLocalStorage(LS_KEY_DOCS, SAMPLE_DOCS);
  const [index, setIndex] = useState(buildTfIdfIndex(docs));
  useEffect(()=>{ setIndex(buildTfIdfIndex(docs)); },[docs]);
  const [showOnboard, setShowOnboard] = useState(!user);
  return (
    <Router basename="/pathfinder">
      <Navbar/>
      <Onboarding open={showOnboard} onClose={()=>setShowOnboard(false)} setUser={setUser}/>
      <Routes>
        <Route path="/" element={<Home user={user} docs={docs}/>}/>
        <Route path="/documents" element={<Documents docs={docs} setDocs={setDocs} setIndex={setIndex}/>}/>
        <Route path="/exams" element={<Exams/>}/>
        <Route path="/career" element={<Career/>}/>
        <Route path="/chat" element={<Chat index={index} docs={docs} user={user}/>}/>
      </Routes>
      <footer className="p-6 text-center text-xs text-gray-500">© {new Date().getFullYear()} Pathfinder AI</footer>
    </Router>
  );
}
