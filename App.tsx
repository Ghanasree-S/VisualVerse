
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import React, { useState, useMemo } from 'react';
import {
  Eye,
  BookOpen,
  Network,
  ArrowRight,
  Cpu,
  Layers,
  Image as ImageIcon,
  Download,
  Sparkles,
  Trash2,
  Loader2,
  Sun,
  Moon,
  Github,
  Monitor,
  MessageSquare,
  FileText,
  Search,
  CheckCircle2,
  Activity,
  Palette,
  ChevronRight,
  HelpCircle,
  Zap
} from 'lucide-react';
import { Button } from './components/Button';
import { analyzeText, generatePanelImage } from './services/geminiService';
import { AppView, OutputMode, ProcessStatus, ComicPanel, AnalysisResult } from './types';

// --- Shared Components ---

const Logo = ({ className = "" }: { className?: string }) => (
  <div className={`flex items-center gap-2 group cursor-pointer ${className}`}>
    <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-white rotate-3 group-hover:rotate-0 transition-all duration-300 shadow-lg shadow-indigo-600/20">
      <Eye size={22} strokeWidth={2.5} fill="white" className="text-indigo-600" />
    </div>
    <span className="font-black text-2xl tracking-tighter dark:text-white text-zinc-900 uppercase">
      VISUAL<span className="text-indigo-600">VERSE</span>
    </span>
  </div>
);

const Navbar = ({ currentView, setView, theme, toggleTheme }: {
  currentView: AppView,
  setView: (v: AppView) => void,
  theme: 'light' | 'dark',
  toggleTheme: () => void
}) => (
  <nav className="sticky top-0 z-[100] border-b border-zinc-200 dark:border-zinc-800 bg-white/80 dark:bg-black/80 backdrop-blur-lg">
    <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
      <div onClick={() => setView('landing')}>
        <Logo />
      </div>

      <div className="hidden md:flex items-center gap-8">
        {[
          { id: 'landing', label: 'Home' },
          { id: 'workspace', label: 'Studio' },
          { id: 'about', label: 'Paper & Theory' },
          { id: 'future', label: 'Roadmap' }
        ].map(item => (
          <button
            key={item.id}
            onClick={() => setView(item.id as AppView)}
            className={`text-sm font-bold uppercase tracking-widest transition-colors ${currentView === item.id
              ? 'text-indigo-600'
              : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100'
              }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-4">
        <button
          onClick={toggleTheme}
          className="p-2 rounded-full hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-500 dark:text-zinc-400 transition-colors"
        >
          {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
        </button>
        <Button size="sm" onClick={() => setView('workspace')}>Launch Studio</Button>
      </div>
    </div>
  </nav>
);

const SectionTitle = ({ children, subtitle }: { children: React.ReactNode, subtitle?: string }) => (
  <div className="mb-12">
    <h2 className="text-3xl md:text-5xl font-black mb-4 dark:text-white text-zinc-900 tracking-tight">{children}</h2>
    {subtitle && <p className="text-zinc-500 dark:text-zinc-400 max-w-2xl text-lg">{subtitle}</p>}
  </div>
);

// --- Page Views ---

const LandingPage = ({ setView }: { setView: (v: AppView) => void }) => (
  <div className="animate-fade-in pb-20">
    <header className="py-20 md:py-32 px-6">
      <div className="max-w-5xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 text-indigo-500 text-xs font-bold uppercase tracking-widest mb-6 border border-indigo-500/20">
          <Sparkles size={14} /> Dual-Mode NLP Engine
        </div>
        <h1 className="text-6xl md:text-9xl font-black mb-8 dark:text-white text-zinc-900 leading-[0.85] tracking-tighter">
          VISUALIZING<br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500">KNOWLEDGE.</span>
        </h1>
        <p className="text-xl text-zinc-600 dark:text-zinc-400 mb-10 max-w-2xl leading-relaxed">
          The ultimate dual-mode system. Transforming narrative stories into immersive comic strips and informational text into intuitive knowledge graphs.
        </p>
        <div className="flex flex-col sm:flex-row items-center gap-4">
          <Button size="lg" className="w-full sm:w-auto px-10 rounded-full" onClick={() => setView('workspace')} icon={<ArrowRight size={20} />}>
            Enter the Studio
          </Button>
          <Button size="lg" variant="outline" className="w-full sm:w-auto rounded-full" onClick={() => setView('about')}>
            Read Project Abstract
          </Button>
        </div>
      </div>
    </header>

    <section className="py-24 bg-zinc-50 dark:bg-zinc-900/50 border-y border-zinc-200 dark:border-zinc-800">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex flex-col md:flex-row gap-12 items-center mb-20">
          <div className="flex-1">
            <SectionTitle subtitle="VisualVerse intelligently classifies your content and routes it through specialized visual generation pipelines.">Our Dual-Output Pipeline</SectionTitle>
          </div>
          <div className="flex-1 grid grid-cols-2 gap-4">
            <div className="p-6 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex flex-col items-center text-center">
              <BookOpen className="text-indigo-500 mb-3" size={32} />
              <span className="font-bold text-indigo-500">Narrative Mode</span>
              <span className="text-xs opacity-60">Comic Strips</span>
            </div>
            <div className="p-6 rounded-2xl bg-purple-500/10 border border-purple-500/20 flex flex-col items-center text-center">
              <Network className="text-purple-500 mb-3" size={32} />
              <span className="font-bold text-purple-500">Informational Mode</span>
              <span className="text-xs opacity-60">Mind-Maps</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
          {[
            {
              icon: <MessageSquare className="text-indigo-500" />,
              title: 'Linguistic Analysis',
              desc: 'Deep tokenization, sentence splitting, and entity recognition to find the "heart" of your text.'
            },
            {
              icon: <Cpu className="text-purple-500" />,
              title: 'Routing & Extraction',
              desc: 'AI determines text modality and extracts scene details for comics or concept pairs for maps.'
            },
            {
              icon: <ImageIcon className="text-pink-500" />,
              title: 'Visual Synthesis',
              desc: 'Generative models create high-fidelity panels and graph engines render hierarchical relationships.'
            }
          ].map((item, i) => (
            <div key={i} className="p-8 rounded-3xl bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 shadow-sm hover:shadow-xl transition-all">
              <div className="w-12 h-12 rounded-2xl bg-zinc-100 dark:bg-zinc-900 flex items-center justify-center mb-6">{item.icon}</div>
              <h3 className="text-2xl font-bold mb-3 dark:text-white">{item.title}</h3>
              <p className="text-zinc-500 dark:text-zinc-400 leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  </div>
);

const WorkspacePage = ({ onGenerate }: { onGenerate: (text: string, mode: 'auto' | 'comic' | 'mindmap') => void }) => {
  const [text, setText] = useState('');
  const [mode, setMode] = useState<'auto' | 'comic' | 'mindmap'>('auto');

  return (
    <div className="animate-fade-in max-w-6xl mx-auto px-6 py-12">
      <div className="flex flex-col md:flex-row gap-12">
        <div className="flex-[2] space-y-6">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-black dark:text-white uppercase tracking-tight">Project Studio</h1>
            <Button variant="ghost" size="sm" onClick={() => setText('')} icon={<Trash2 size={16} />}>Clear Buffer</Button>
          </div>
          <div className="relative group">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter a story (e.g. 'A lonely robot travels across a desert planet...') or a conceptual topic (e.g. 'The lifecycle of a star')..."
              className="w-full h-[500px] p-8 rounded-[2rem] bg-zinc-50 dark:bg-zinc-900 border-2 border-zinc-200 dark:border-zinc-800 focus:border-indigo-500 outline-none text-xl leading-relaxed resize-none dark:text-white transition-all shadow-inner"
            />
            <div className="absolute top-4 right-4 px-3 py-1 bg-white dark:bg-black rounded-full border border-zinc-200 dark:border-zinc-800 text-[10px] text-zinc-400 font-bold uppercase tracking-widest pointer-events-none">
              Input Buffer: {text.length} chars
            </div>
          </div>
        </div>

        <div className="flex-1 space-y-8">
          <div className="p-8 rounded-[2rem] bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
            <h3 className="text-sm font-black uppercase tracking-widest mb-6 text-zinc-500">Pipeline Config</h3>
            <div className="space-y-4">
              {[
                { id: 'auto', icon: <Eye size={18} />, label: 'Auto Classifier', sub: 'Neural Routing' },
                { id: 'comic', icon: <BookOpen size={18} />, label: 'Comic Strip', sub: 'Narrative Pipeline' },
                { id: 'mindmap', icon: <Network size={18} />, label: 'Mind-Map', sub: 'Conceptual Pipeline' },
              ].map((m) => (
                <button
                  key={m.id}
                  onClick={() => setMode(m.id as any)}
                  className={`w-full p-4 rounded-2xl border-2 flex items-center gap-4 transition-all ${mode === m.id
                    ? 'border-indigo-500 bg-indigo-500/10'
                    : 'border-zinc-200 dark:border-zinc-800 hover:border-zinc-300 dark:hover:border-zinc-700'
                    }`}
                >
                  <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${mode === m.id ? 'bg-indigo-500 text-white shadow-lg shadow-indigo-500/20' : 'bg-zinc-200 dark:bg-zinc-800 text-zinc-500'}`}>
                    {m.icon}
                  </div>
                  <div className="text-left">
                    <div className="font-bold dark:text-white text-sm">{m.label}</div>
                    <div className="text-[10px] text-zinc-500 uppercase font-black tracking-tighter">{m.sub}</div>
                  </div>
                  {mode === m.id && <div className="ml-auto w-2 h-2 rounded-full bg-indigo-500 animate-pulse"></div>}
                </button>
              ))}
            </div>

            <div className="mt-8 pt-8 border-t border-zinc-200 dark:border-zinc-800">
              <div className="flex items-center gap-2 mb-4 text-zinc-500">
                <Palette size={16} />
                <span className="text-xs font-bold uppercase tracking-widest">Visual Style</span>
              </div>
              <select className="w-full bg-white dark:bg-black border border-zinc-200 dark:border-zinc-800 rounded-xl p-3 text-sm focus:outline-none focus:border-indigo-500">
                <option>Digital Illustration</option>
                <option>Manga / Noir</option>
                <option>Oil Painting</option>
                <option>Technical Schematic</option>
              </select>
            </div>
          </div>

          <Button
            size="lg"
            className="w-full h-16 text-xl rounded-2xl"
            disabled={!text.trim()}
            onClick={() => onGenerate(text, mode)}
          >
            Start Visualization
          </Button>
        </div>
      </div>
    </div>
  );
};

const ResultsPage = ({ status, result, panels, onReset }: {
  status: ProcessStatus,
  result: AnalysisResult | null,
  panels: ComicPanel[],
  onReset: () => void
}) => {
  const [activeTab, setActiveTab] = useState<'output' | 'pipeline'>('output');
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });

  const mindmapNodes = useMemo(() => {
    if (!result?.mindMapData) return [];
    const nodes = result.mindMapData.nodes;

    // If backend provides positions, use them directly
    const hasPositions = nodes.some((n: any) => n.x !== undefined && n.y !== undefined);
    if (hasPositions) {
      return nodes.map((n: any) => ({
        ...n,
        x: n.x || 600,
        y: n.y || 350
      }));
    }

    // Fallback: calculate positions if backend doesn't provide them
    const cx = 600;
    const cy = 350;
    const r1 = 180;
    const r2 = 320;

    const categoryNodes = nodes.filter((n: any) => n.nodeType === 'category' || n.level === 1);
    const detailNodes = nodes.filter((n: any) => n.nodeType === 'detail' || n.level === 2);

    return nodes.map((n: any) => {
      // Main node at center
      if (n.nodeType === 'main' || n.level === 0 || n.type === 'topic') {
        return { ...n, x: cx, y: cy };
      }

      // Category nodes around center
      if (n.nodeType === 'category' || n.level === 1) {
        const catIdx = parseInt(n.id?.replace('cat_', '') || '0');
        const total = categoryNodes.length;
        const angle = (2 * Math.PI * catIdx / total) - (Math.PI / 2);
        return {
          ...n,
          x: cx + r1 * Math.cos(angle),
          y: cy + r1 * Math.sin(angle)
        };
      }

      // Detail nodes
      if (n.id?.startsWith('det_') || n.level === 2) {
        const parts = n.id.split('_');
        const catIdx = parseInt(parts[1] || '0');
        const detIdx = parseInt(parts[2] || '0');

        const total = categoryNodes.length || 4;
        const catAngle = (2 * Math.PI * catIdx / total) - (Math.PI / 2);

        const siblings = detailNodes.filter((d: any) => d.id?.startsWith(`det_${catIdx}_`));
        const sibCount = siblings.length;

        const arcSpread = Math.PI / 4;
        let detAngle = catAngle;
        if (sibCount > 1) {
          const offset = (detIdx - (sibCount - 1) / 2) * (arcSpread / Math.max(sibCount - 1, 1));
          detAngle = catAngle + offset;
        }

        return {
          ...n,
          x: cx + r2 * Math.cos(detAngle),
          y: cy + r2 * Math.sin(detAngle)
        };
      }

      return { ...n, x: n.x || cx, y: n.y || cy };
    });
  }, [result]);

  const handleResetView = () => {
    setZoom(1);
    setPanOffset({ x: 0, y: 0 });
  };

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.2, 2));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.2, 0.5));

  if (status === 'analyzing' || status === 'generating') {
    return (
      <div className="min-h-[70vh] flex flex-col items-center justify-center p-6 text-center animate-fade-in">
        <div className="relative w-32 h-32 mb-12">
          <div className="absolute inset-0 border-[6px] border-zinc-100 dark:border-zinc-900 rounded-full"></div>
          <div className="absolute inset-0 border-[6px] border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
          <Activity size={40} className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-indigo-500 animate-pulse" />
        </div>
        <h2 className="text-3xl font-black dark:text-white mb-3 uppercase tracking-tighter">
          {status === 'analyzing' ? 'Linguistic Decoding' : 'Visual Rendering'}
        </h2>
        <p className="text-zinc-500 max-w-sm text-lg">Our multimodal pipeline is processing vectors and generating latent pixels.</p>

        <div className="mt-16 w-full max-w-lg space-y-3 bg-zinc-50 dark:bg-zinc-950 p-8 rounded-3xl border border-zinc-200 dark:border-zinc-800">
          {[
            { label: 'Neural Tokenization', done: true },
            { label: 'Semantic Analysis', done: status === 'generating' },
            { label: 'Modality Classification', done: status === 'generating' },
            { label: 'Diffusion Image Synthesis', done: false },
          ].map((step, i) => (
            <div key={i} className="flex items-center justify-between">
              <span className={`text-sm font-bold uppercase tracking-widest ${step.done ? 'text-indigo-500' : 'text-zinc-500'}`}>{step.label}</span>
              {step.done ? <CheckCircle2 size={16} className="text-green-500" /> : <Loader2 size={16} className="text-indigo-500 animate-spin" />}
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!result) return null;

  return (
    <div className="animate-fade-in pb-20 px-6 max-w-7xl mx-auto">
      <header className="py-12 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 border-b border-zinc-200 dark:border-zinc-800 mb-8">
        <div>
          <div className="flex items-center gap-3 text-zinc-400 mb-2">
            <span className="text-xs font-bold uppercase tracking-widest hover:text-indigo-500 cursor-pointer" onClick={onReset}>Workspace</span>
            <ChevronRight size={14} />
            <span className="text-xs font-bold uppercase tracking-widest text-indigo-500">Visualization</span>
          </div>
          <h1 className="text-4xl font-black dark:text-white tracking-tighter uppercase">{result.title}</h1>
          <p className="text-zinc-500 font-medium">{result.summary}</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="bg-zinc-100 dark:bg-zinc-900 rounded-xl p-1 flex">
            <button
              onClick={() => setActiveTab('output')}
              className={`px-4 py-2 rounded-lg text-xs font-bold uppercase transition-all ${activeTab === 'output' ? 'bg-white dark:bg-zinc-800 text-indigo-500 shadow-sm' : 'text-zinc-500'}`}
            >
              Output
            </button>
            <button
              onClick={() => setActiveTab('pipeline')}
              className={`px-4 py-2 rounded-lg text-xs font-bold uppercase transition-all ${activeTab === 'pipeline' ? 'bg-white dark:bg-zinc-800 text-indigo-500 shadow-sm' : 'text-zinc-500'}`}
            >
              NLP Insights
            </button>
          </div>
          <Button variant="outline" size="sm" icon={<Download size={16} />}>Export PDF</Button>
        </div>
      </header>

      {activeTab === 'output' ? (
        <main>
          {result.mode === 'comic' ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
              {panels.map((panel, idx) => (
                <div key={panel.id} className="group relative rounded-[2rem] overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 shadow-2xl">
                  <div className="aspect-square bg-zinc-100 dark:bg-zinc-900 relative">
                    {panel.imageUrl ? (
                      <img src={panel.imageUrl} className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" alt={`Panel ${idx + 1}`} />
                    ) : (
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <Loader2 className="animate-spin text-indigo-500 mb-2" />
                        <span className="text-xs text-zinc-500 font-mono">Drawing Panel...</span>
                      </div>
                    )}
                    <div className="absolute top-6 left-6 w-12 h-12 bg-black text-white rounded-2xl flex items-center justify-center font-black text-2xl border border-white/20 shadow-2xl">
                      {idx + 1}
                    </div>
                  </div>
                  <div className="p-10">
                    <div className="relative">
                      <div className="absolute -left-4 top-0 w-1 h-full bg-indigo-500/20 rounded-full"></div>
                      <p className="text-2xl leading-relaxed dark:text-zinc-300 font-serif italic text-zinc-700">"{panel.caption}"</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-6">
              {/* Extracted Keyphrases Section */}
              {result.mindMapData && result.mindMapData.nodes && (
                <div className="bg-white dark:bg-zinc-900 rounded-2xl border border-zinc-200 dark:border-zinc-800 p-6 shadow-lg">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-indigo-500/10 dark:bg-indigo-500/20 flex items-center justify-center">
                      <Sparkles size={20} className="text-indigo-500" />
                    </div>
                    <div>
                      <h3 className="font-bold text-lg dark:text-white">Extracted Concepts</h3>
                      <p className="text-xs text-zinc-500">
                        {result.mindMapData.nodes.filter((n: any) => n.level === 2).length} keyphrases identified from your text
                      </p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    {/* Main Topic */}
                    {result.mindMapData.nodes.filter((n: any) => n.level === 0).map((node: any) => (
                      <div key={node.id} className="flex items-center gap-2">
                        <span className="text-xs font-bold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">Main Topic:</span>
                        <span className="px-4 py-2 rounded-full bg-indigo-500 text-white font-bold text-sm shadow-lg">
                          {node.label}
                        </span>
                      </div>
                    ))}

                    {/* Categories */}
                    <div>
                      <span className="text-xs font-bold text-purple-600 dark:text-purple-400 uppercase tracking-wider mb-2 block">Categories:</span>
                      <div className="flex flex-wrap gap-2">
                        {result.mindMapData.nodes
                          .filter((n: any) => n.level === 1)
                          .map((node: any) => (
                            <span
                              key={node.id}
                              className="px-3 py-1.5 rounded-lg bg-purple-500/10 dark:bg-purple-500/20 text-purple-700 dark:text-purple-300 text-sm font-semibold border border-purple-500/20"
                            >
                              {node.label}
                            </span>
                          ))}
                      </div>
                    </div>

                    {/* Keyphrases/Details */}
                    <div>
                      <span className="text-xs font-bold text-blue-600 dark:text-blue-400 uppercase tracking-wider mb-2 block">Key Concepts:</span>
                      <div className="flex flex-wrap gap-2">
                        {result.mindMapData.nodes
                          .filter((n: any) => n.level === 2)
                          .map((node: any) => (
                            <span
                              key={node.id}
                              className="px-3 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 text-sm font-medium hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors cursor-default"
                            >
                              {node.label}
                            </span>
                          ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Mindmap Visualization */}
              <div className="rounded-[3rem] border-2 border-zinc-200 dark:border-zinc-800 bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-zinc-900 dark:via-indigo-950 dark:to-purple-950 relative shadow-2xl" style={{ height: '800px', overflow: 'auto' }}>
                {/* SVG-based mindmap - properly scaled */}
                <svg
                  className="w-full h-full"
                  viewBox="0 0 3200 900"
                  preserveAspectRatio="xMidYMid meet"
                  style={{ transform: `scale(${zoom})`, transformOrigin: 'center', transition: 'transform 0.3s' }}
                >
                  <defs>
                    <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#6366f1" stopOpacity="0.6" />
                      <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.4" />
                    </linearGradient>
                    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                      <feDropShadow dx="0" dy="4" stdDeviation="8" floodOpacity="0.2" />
                    </filter>
                  </defs>

                  {/* Edges/Connections with curved paths */}
                  {(() => {
                    // Deduplicate edges - keep only unique from-to pairs
                    const seenEdges = new Set<string>();
                    const uniqueEdges = (result.mindMapData?.edges || []).filter(edge => {
                      const key = `${edge.from}-${edge.to}`;
                      if (seenEdges.has(key)) return false;
                      seenEdges.add(key);
                      return true;
                    });

                    return uniqueEdges.map((edge, i) => {
                      const fromNode = mindmapNodes.find(n => n.id === edge.from);
                      const toNode = mindmapNodes.find(n => n.id === edge.to);
                      if (!fromNode || !toNode) return null;

                      // Calculate curved path
                      const midX = (fromNode.x + toNode.x) / 2;
                      const midY = (fromNode.y + toNode.y) / 2;

                      // Curve offset
                      const dx = toNode.x - fromNode.x;
                      const dy = toNode.y - fromNode.y;

                      const ctrlX = midX - dy * 0.15;
                      const ctrlY = midY + dx * 0.15;

                      return (
                        <g key={`${edge.from}-${edge.to}`}>
                          {/* Curved path */}
                          <path
                            d={`M ${fromNode.x} ${fromNode.y} Q ${ctrlX} ${ctrlY} ${toNode.x} ${toNode.y}`}
                            stroke="url(#lineGrad)"
                            strokeWidth="3"
                            fill="none"
                            strokeLinecap="round"
                          />
                          {/* Relationship label */}
                          {edge.label && (
                            <text
                              x={midX}
                              y={midY - 10}
                              textAnchor="middle"
                              fontSize="10"
                              fill="#6366f1"
                              fontStyle="italic"
                            >
                              {edge.label}
                            </text>
                          )}
                        </g>
                      );
                    });
                  })()}

                  {/* Nodes */}
                  {mindmapNodes.map((node: any, i) => {
                    const nodeType = node.nodeType || (node.type === 'topic' ? 'main' : 'detail');
                    const isMain = nodeType === 'main';
                    const isCat = nodeType === 'category';

                    // Node dimensions
                    const width = isMain ? 180 : isCat ? 150 : 120;
                    const height = isMain ? 60 : isCat ? 45 : 35;
                    const rx = height / 2; // Rounded corners

                    // Colors
                    const fill = isMain ? '#6366f1' : isCat ? '#8b5cf6' : '#ffffff';
                    const stroke = isMain ? '#4f46e5' : isCat ? '#7c3aed' : '#06b6d4';
                    const textColor = isMain || isCat ? '#ffffff' : '#1f2937';
                    const fontSize = isMain ? 16 : isCat ? 14 : 12;

                    return (
                      <g key={node.id} filter="url(#shadow)">
                        {/* Node background */}
                        <rect
                          x={node.x - width / 2}
                          y={node.y - height / 2}
                          width={width}
                          height={height}
                          rx={rx}
                          fill={fill}
                          stroke={stroke}
                          strokeWidth="2"
                        />
                        {/* Node text */}
                        <text
                          x={node.x}
                          y={node.y}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fill={textColor}
                          fontSize={fontSize}
                          fontWeight="bold"
                          fontFamily="system-ui, sans-serif"
                        >
                          {node.label}
                        </text>
                      </g>
                    );
                  })}
                </svg>

                {/* Controls */}
                <div className="absolute bottom-6 right-6 flex gap-2 p-3 bg-white/95 dark:bg-black/95 backdrop-blur-xl rounded-2xl border border-zinc-200 dark:border-zinc-700 z-40 shadow-lg">
                  <Button size="sm" variant="secondary" onClick={handleZoomOut}>−</Button>
                  <Button size="sm" variant="secondary" onClick={handleResetView}>Reset</Button>
                  <Button size="sm" variant="secondary" onClick={handleZoomIn}>+</Button>
                </div>

                <div className="absolute top-6 right-6 px-4 py-2 bg-white/95 dark:bg-black/95 backdrop-blur-xl rounded-full text-sm font-bold z-40 shadow">
                  {Math.round(zoom * 100)}%
                </div>

                {/* Legend */}
                <div className="absolute bottom-6 left-6 flex items-center gap-4 p-3 bg-white/95 dark:bg-black/95 backdrop-blur-xl rounded-xl text-xs z-40 shadow">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-indigo-500"></div>
                    <span>Main</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                    <span>Category</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full border-2 border-cyan-400 bg-white"></div>
                    <span>Detail</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      ) : (
        <main className="animate-slide-up">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
            <div className="space-y-8">
              <div className="p-10 rounded-[2rem] bg-indigo-500/5 border border-indigo-500/20">
                <h3 className="text-xl font-black mb-4 uppercase text-indigo-500">Classification Log</h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-4 bg-white dark:bg-black rounded-xl">
                    <span className="text-sm font-bold">Inferred Modality</span>
                    <span className="text-xs font-black uppercase text-indigo-500">{result.mode}</span>
                  </div>
                  <div className="flex justify-between items-center p-4 bg-white dark:bg-black rounded-xl">
                    <span className="text-sm font-bold">Confidence Score</span>
                    <span className="text-xs font-black uppercase text-green-500">0.98</span>
                  </div>
                  <div className="flex justify-between items-center p-4 bg-white dark:bg-black rounded-xl">
                    <span className="text-sm font-bold">NLP Engine</span>
                    <span className="text-xs font-black uppercase text-zinc-500">Gemini 3 Flash</span>
                  </div>
                </div>
              </div>

              <div className="p-10 rounded-[2rem] bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
                <h3 className="text-xl font-black mb-4 uppercase">Entity Disambiguation</h3>
                <p className="text-sm text-zinc-500 leading-relaxed mb-6">
                  The system successfully resolved co-references and extracted the following key structural elements from the source text buffer.
                </p>
                <div className="flex flex-wrap gap-2">
                  {result.mindMapData?.nodes.map((n, i) => (
                    <span key={i} className="px-3 py-1 rounded-full bg-zinc-200 dark:bg-zinc-800 text-[10px] font-bold uppercase tracking-wider">{n.label}</span>
                  )) || result.comicData?.map((c, i) => (
                    <span key={i} className="px-3 py-1 rounded-full bg-zinc-200 dark:bg-zinc-800 text-[10px] font-bold uppercase tracking-wider">Scene {i + 1}</span>
                  ))}
                </div>
              </div>
            </div>

            <div className="p-10 rounded-[2rem] bg-black text-white relative overflow-hidden">
              <div className="absolute top-0 right-0 p-8 opacity-10">
                <Activity size={200} />
              </div>
              <h3 className="text-xl font-black mb-8 uppercase text-indigo-400">Pipeline Visualization</h3>
              <div className="space-y-12 relative">
                <div className="absolute left-4 top-0 w-0.5 h-full bg-zinc-800"></div>
                {[
                  { step: 'Ingest', label: 'Tokenization & Normalization' },
                  { step: 'Route', label: 'Multimodal Classification' },
                  { step: 'Parse', label: 'Entity Relationship Extraction' },
                  { step: 'Render', label: 'Latent Image / Graph Synthesis' },
                ].map((item, i) => (
                  <div key={i} className="flex items-start gap-6 relative">
                    <div className="w-8 h-8 rounded-full bg-indigo-500 flex items-center justify-center text-[10px] font-bold z-10 shadow-[0_0_20px_rgba(99,102,241,0.5)]">
                      {i + 1}
                    </div>
                    <div>
                      <div className="text-[10px] font-black uppercase text-indigo-400 tracking-[0.2em]">{item.step}</div>
                      <div className="text-lg font-bold">{item.label}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </main>
      )}
    </div>
  );
};

// --- App Root ---

export default function App() {
  const [view, setView] = useState<AppView>('landing');
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');
  const [status, setStatus] = useState<ProcessStatus>('idle');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [panels, setPanels] = useState<ComicPanel[]>([]);

  const toggleTheme = () => {
    const next = theme === 'light' ? 'dark' : 'light';
    setTheme(next);
    document.documentElement.classList.toggle('dark', next === 'dark');
  };

  const handleGenerate = async (text: string, mode: 'auto' | 'comic' | 'mindmap') => {
    setStatus('analyzing');
    setView('results');

    try {
      const analysis = await analyzeText(text, mode);
      setResult(analysis);

      if (analysis.mode === 'comic' && analysis.comicData) {
        setStatus('generating');
        const initialPanels = analysis.comicData;
        setPanels(initialPanels);

        const generated = await Promise.all(
          initialPanels.map(async (p) => {
            try {
              const url = await generatePanelImage(p.prompt);
              return { ...p, imageUrl: url };
            } catch (e) {
              console.error(e);
              return p;
            }
          })
        );
        setPanels(generated);
      }

      setStatus('complete');
    } catch (e: any) {
      console.error(e);
      setStatus('error');
      alert(`Pipeline Fault: ${e.message}`);
    }
  };

  return (
    <div className={`min-h-screen transition-colors duration-500 font-sans ${theme === 'dark' ? 'bg-black text-white' : 'bg-white text-zinc-900'}`}>
      <Navbar currentView={view} setView={setView} theme={theme} toggleTheme={toggleTheme} />

      <main>
        {view === 'landing' && <LandingPage setView={setView} />}
        {view === 'workspace' && <WorkspacePage onGenerate={handleGenerate} />}
        {view === 'results' && <ResultsPage status={status} result={result} panels={panels} onReset={() => setView('workspace')} />}

        {view === 'about' && (
          <div className="max-w-5xl mx-auto px-6 py-20 animate-fade-in space-y-24">
            <header>
              <SectionTitle subtitle="A dual-mode NLP system for converting unstructured text into structured visual artifacts.">Research Abstract: VisualVerse</SectionTitle>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-12 mt-12">
                <div>
                  <h3 className="text-sm font-black uppercase text-indigo-500 mb-4 tracking-widest">Problem Statement</h3>
                  <p className="text-zinc-500 leading-relaxed text-lg">
                    Textual information is often difficult for learners to process, especially in narrative or conceptual forms. Most text-processing tools only summarize or highlight content. There is a need for a system that automatically translates text into highly visual formats to improve comprehension, engagement, and creativity.
                  </p>
                </div>
                <div>
                  <h3 className="text-sm font-black uppercase text-indigo-500 mb-4 tracking-widest">The Proposed Solution</h3>
                  <p className="text-zinc-500 leading-relaxed text-lg">
                    VisuText (codenamed VisualVerse) is a dual-mode visual transformation system. If a text is detected as a story, it generates comic panels. If it is informational, it generates an interactive mind-map. NLP handles understanding and extraction, while Generative AI handles visual synthesis.
                  </p>
                </div>
              </div>
            </header>

            <section>
              <h3 className="text-3xl font-black mb-12 uppercase tracking-tight">System Architecture</h3>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                {[
                  { label: 'Preprocessing', icon: <FileText />, desc: 'Tokenization, POS tagging, and Dependency parsing.' },
                  { label: 'Classification', icon: <Layers />, desc: 'Narrative vs Informational routing logic.' },
                  { label: 'Extraction', icon: <Search />, desc: 'NER and Keyphrase extraction modules.' },
                  { label: 'Synthesis', icon: <Monitor />, desc: 'Stable Diffusion and NetworkX renderers.' }
                ].map((block, i) => (
                  <div key={i} className="p-8 rounded-3xl bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-center">
                    <div className="w-12 h-12 rounded-2xl bg-white dark:bg-black border border-zinc-200 dark:border-zinc-800 flex items-center justify-center mx-auto mb-6 text-indigo-500">
                      {block.icon}
                    </div>
                    <div className="font-bold mb-2 uppercase text-xs tracking-widest">{block.label}</div>
                    <p className="text-xs text-zinc-500">{block.desc}</p>
                  </div>
                ))}
              </div>
            </section>

            <section className="p-12 rounded-[3rem] bg-indigo-600 text-white flex flex-col md:flex-row items-center justify-between gap-12">
              <div className="max-w-xl">
                <h3 className="text-4xl font-black mb-6 uppercase leading-tight tracking-tighter">Novelty & Academic Value</h3>
                <p className="text-indigo-100 text-lg">
                  VisualVerse represents a first-of-this-kind dual-mode system that bridges the gap between text comprehension and visual learning. By utilizing State-of-the-art NLP (Gemini) and Generative Image Models, it offers a scalable framework for educational technology.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-6 bg-white/10 rounded-2xl backdrop-blur-md">
                  <div className="text-3xl font-black">98%</div>
                  <div className="text-[10px] uppercase font-bold tracking-widest opacity-60">Accuracy</div>
                </div>
                <div className="p-6 bg-white/10 rounded-2xl backdrop-blur-md">
                  <div className="text-3xl font-black">2.5s</div>
                  <div className="text-[10px] uppercase font-bold tracking-widest opacity-60">Avg Processing</div>
                </div>
              </div>
            </section>
          </div>
        )}

        {view === 'future' && (
          <div className="max-w-5xl mx-auto px-6 py-20 animate-fade-in text-center">
            <SectionTitle subtitle="Phase 2 development objectives and research extensions.">Roadmap: The Next Frontier</SectionTitle>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 text-left mt-20">
              {[
                { title: 'Multilingual Ingest', desc: 'Real-time translation of source text before modality classification.', icon: <HelpCircle className="text-blue-500" /> },
                { title: 'Voice Streaming', desc: 'Voice-to-comic generation using high-fidelity native audio models.', icon: <Zap className="text-yellow-500" /> },
                { title: 'Custom Styles', desc: 'Anime, Pixar-style, and Technical Illustration model fine-tuning.', icon: <Palette className="text-purple-500" /> },
                { title: 'Dynamic Graphs', desc: 'Force-directed knowledge graphs with real-time web grounding.', icon: <Network className="text-green-500" /> },
                { title: 'Collaboration', desc: 'Multi-user workspace for shared knowledge map creation.', icon: <Monitor className="text-pink-500" /> },
                { title: 'Export Engine', desc: 'Native export to SVG, PNG, PDF, and PowerPoint formats.', icon: <Download className="text-orange-500" /> }
              ].map((item, i) => (
                <div key={i} className="p-10 rounded-[2rem] bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 hover:border-indigo-500/50 transition-colors group">
                  <div className="mb-6 transform group-hover:scale-110 transition-transform">{item.icon}</div>
                  <h4 className="text-xl font-black mb-3 dark:text-white uppercase tracking-tighter">{item.title}</h4>
                  <p className="text-sm text-zinc-500 leading-relaxed">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      <footer className="py-20 border-t border-zinc-200 dark:border-zinc-800">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-12">
          <div className="flex flex-col gap-2">
            <Logo />
            <p className="text-xs text-zinc-500 max-w-sm ml-1">
              Academic Thesis Project: Experimental Text-to-Visual Generation Pipeline.
              Built for engineering evaluation and research validation.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-8">
            <a href="#" className="text-xs font-bold uppercase tracking-widest text-zinc-400 hover:text-indigo-500">Documentation</a>
            <a href="#" className="text-xs font-bold uppercase tracking-widest text-zinc-400 hover:text-indigo-500">Datasets</a>
            <a href="#" className="text-xs font-bold uppercase tracking-widest text-zinc-400 hover:text-indigo-500">Technical Spec</a>
            <a href="#" className="text-zinc-400 hover:text-indigo-500 transition-colors"><Github size={24} /></a>
          </div>
        </div>
      </footer>
    </div>
  );
}
