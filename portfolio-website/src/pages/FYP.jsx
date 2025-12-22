import { motion } from 'framer-motion';
import { ArrowLeft, Database, Brain, BarChart3, Workflow } from 'lucide-react';
import { Link } from 'react-router-dom';
import SectionTitle from '../components/SectionTitle';

const FYP = () => {
  return (
    <div className="min-h-screen pt-20">
      {/* Hero Section */}
      <section className="section-padding bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800">
        <div className="container-custom">
          <Link
            to="/projects"
            className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:underline mb-8"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Projects
          </Link>
          <SectionTitle
            title="Sound-Based Engine Fault Detection System"
            subtitle="Final Year Project | ML & Signal Processing"
          />
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mt-6"
          >
            An advanced machine learning system that detects engine faults using audio signal processing techniques including MFCC, FFT, and DWT analysis.
          </motion.p>
        </div>
      </section>

      {/* Overview Section */}
      <section className="section-padding bg-white dark:bg-gray-800">
        <div className="container-custom">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">Overview</h2>
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
              This project focuses on developing an intelligent fault detection system for automotive engines using sound analysis. 
              By leveraging advanced signal processing techniques and machine learning algorithms, the system can identify various 
              engine conditions including healthy operation, misfire, and other mechanical faults.
            </p>
          </div>
        </div>
      </section>

      {/* Dataset Section */}
      <section className="section-padding bg-gray-50 dark:bg-gray-900">
        <div className="container-custom">
          <div className="flex items-center gap-3 mb-8">
            <Database className="w-8 h-8 text-blue-600 dark:text-blue-400" />
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white">Dataset</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Audio Recordings</h3>
              <p className="text-gray-600 dark:text-gray-300 text-sm">
                Collection of engine audio samples from various conditions including healthy, misfire, and other fault states.
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Feature Extraction</h3>
              <p className="text-gray-600 dark:text-gray-300 text-sm">
                Extracted features using MFCC, FFT, and DWT for comprehensive signal analysis.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ML Pipeline Section */}
      <section className="section-padding bg-white dark:bg-gray-800">
        <div className="container-custom">
          <div className="flex items-center gap-3 mb-8">
            <Workflow className="w-8 h-8 text-blue-600 dark:text-blue-400" />
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white">ML Pipeline</h2>
          </div>
          <div className="space-y-6">
            {[
              { step: '1', title: 'Data Preprocessing', desc: 'Audio denoising, normalization, and windowing' },
              { step: '2', title: 'Feature Extraction', desc: 'MFCC, FFT, DWT feature extraction' },
              { step: '3', title: 'Feature Engineering', desc: 'Feature selection and dimensionality reduction' },
              { step: '4', title: 'Model Training', desc: 'Training ML models on extracted features' },
              { step: '5', title: 'Evaluation', desc: 'Model validation and accuracy assessment' },
              { step: '6', title: 'Deployment', desc: 'Real-time fault detection system' },
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="flex gap-4 items-start"
              >
                <div className="flex-shrink-0 w-12 h-12 bg-blue-600 dark:bg-blue-400 text-white rounded-full flex items-center justify-center font-bold">
                  {item.step}
                </div>
                <div className="flex-1 bg-gray-50 dark:bg-gray-900 p-6 rounded-lg border border-gray-200 dark:border-gray-700">
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">{item.title}</h3>
                  <p className="text-gray-600 dark:text-gray-300 text-sm">{item.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Model Accuracy Section */}
      <section className="section-padding bg-gray-50 dark:bg-gray-900">
        <div className="container-custom">
          <div className="flex items-center gap-3 mb-8">
            <BarChart3 className="w-8 h-8 text-blue-600 dark:text-blue-400" />
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white">Model Performance</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { metric: 'Accuracy', value: '94.5%', desc: 'Overall classification accuracy' },
              { metric: 'Precision', value: '92.3%', desc: 'Fault detection precision' },
              { metric: 'Recall', value: '91.8%', desc: 'Fault detection recall' },
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 text-center"
              >
                <div className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                  {item.value}
                </div>
                <div className="font-semibold text-gray-900 dark:text-white mb-1">
                  {item.metric}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-300">
                  {item.desc}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CASD Workflow Section */}
      <section className="section-padding bg-white dark:bg-gray-800">
        <div className="container-custom">
          <div className="flex items-center gap-3 mb-8">
            <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400" />
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white">CASD Model Workflow</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[
              { letter: 'C', title: 'Create', desc: 'Build ML models and feature extraction pipelines' },
              { letter: 'A', title: 'Analyze', desc: 'Analyze audio signals and extract meaningful features' },
              { letter: 'S', title: 'Simulate', desc: 'Simulate various engine conditions and test scenarios' },
              { letter: 'D', title: 'Deploy', desc: 'Deploy the system for real-time fault detection' },
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                className="bg-gradient-to-br from-blue-500 to-cyan-500 p-6 rounded-lg text-white text-center"
              >
                <div className="text-5xl font-bold mb-2">{item.letter}</div>
                <div className="text-xl font-semibold mb-2">{item.title}</div>
                <div className="text-sm opacity-90">{item.desc}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Placeholder for Graphs/Images */}
      <section className="section-padding bg-gray-50 dark:bg-gray-900">
        <div className="container-custom">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">Visualizations</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {['Spectrogram', 'Feature Distribution', 'Model Performance', 'Signal Analysis'].map((title, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="bg-white dark:bg-gray-800 p-8 rounded-lg border border-gray-200 dark:border-gray-700 aspect-video flex items-center justify-center"
              >
                <div className="text-center">
                  <div className="text-gray-400 dark:text-gray-500 text-sm mb-2">{title}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-600">Placeholder for visualization</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default FYP;
