import { motion } from 'framer-motion';
import { ArrowLeft, Grip, Ruler, Target, Zap } from 'lucide-react';
import { Link } from 'react-router-dom';
import SectionTitle from '../components/SectionTitle';

const Cozmoclench = () => {
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
            title="Cozmoclench Bot - Gripper Mechanism"
            subtitle="Robotics | Mechanism Design"
          />
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mt-6"
          >
            An innovative gripper design for robotics competition with strict size constraints (300×200×300 mm) 
            and high precision requirements for object manipulation.
          </motion.p>
        </div>
      </section>

      {/* Overview Section */}
      <section className="section-padding bg-white dark:bg-gray-800">
        <div className="container-custom">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">Project Overview</h2>
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed max-w-3xl">
            The Cozmoclench Bot is a specialized gripper mechanism designed for robotics competitions. 
            The project required innovative design solutions to maximize gripping capability while adhering to 
            strict dimensional constraints. The mechanism demonstrates advanced understanding of kinematics, 
            force analysis, and material selection.
          </p>
        </div>
      </section>

      {/* Key Features */}
      <section className="section-padding bg-gray-50 dark:bg-gray-900">
        <div className="container-custom">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center gap-3 mb-4">
                <Grip className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                  Gripper Design
                </h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed mb-4">
                Designed a multi-fingered gripper with adaptive grasping capability. The mechanism uses 
                a combination of parallel and angular jaw movements to handle objects of various shapes and sizes.
              </p>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 dark:text-blue-400 mt-1">•</span>
                  <span>Multi-fingered configuration for stability</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 dark:text-blue-400 mt-1">•</span>
                  <span>Adaptive finger positioning</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 dark:text-blue-400 mt-1">•</span>
                  <span>High gripping force-to-weight ratio</span>
                </li>
              </ul>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center gap-3 mb-4">
                <Ruler className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                  Size Constraints
                </h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed mb-4">
                The mechanism had to fit within strict dimensional limits while maintaining functionality:
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                <div className="text-sm text-gray-600 dark:text-gray-300 space-y-1">
                  <div><strong>Width:</strong> 300 mm</div>
                  <div><strong>Depth:</strong> 200 mm</div>
                  <div><strong>Height:</strong> 300 mm</div>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center gap-3 mb-4">
                <Target className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                  Mechanism Explanation
                </h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed">
                The gripper uses a four-bar linkage mechanism combined with a rack-and-pinion system for 
                precise finger control. The design incorporates a single actuator that drives multiple fingers 
                simultaneously, ensuring synchronized movement and consistent gripping force distribution.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center gap-3 mb-4">
                <Zap className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                  Technical Highlights
                </h3>
              </div>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 dark:text-blue-400 mt-1">•</span>
                  <span>Kinematic analysis and optimization</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 dark:text-blue-400 mt-1">•</span>
                  <span>Force analysis and material selection</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 dark:text-blue-400 mt-1">•</span>
                  <span>CAD modeling and prototyping</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 dark:text-blue-400 mt-1">•</span>
                  <span>Testing and validation</span>
                </li>
              </ul>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Design Process */}
      <section className="section-padding bg-white dark:bg-gray-800">
        <div className="container-custom">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">Design Process</h2>
          <div className="space-y-6">
            {[
              { step: '1', title: 'Requirements Analysis', desc: 'Analyzed competition requirements and constraints' },
              { step: '2', title: 'Conceptual Design', desc: 'Developed multiple design concepts and evaluated trade-offs' },
              { step: '3', title: 'Kinematic Analysis', desc: 'Performed kinematic analysis to optimize mechanism performance' },
              { step: '4', title: 'CAD Modeling', desc: 'Created detailed 3D models using SolidWorks' },
              { step: '5', title: 'Prototyping', desc: 'Built and tested physical prototypes' },
              { step: '6', title: 'Optimization', desc: 'Iterated design based on testing results' },
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

      {/* Placeholder for Images */}
      <section className="section-padding bg-gray-50 dark:bg-gray-900">
        <div className="container-custom">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">Design Gallery</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {['CAD Model', 'Mechanism Detail', 'Final Assembly'].map((title, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                className="bg-white dark:bg-gray-800 aspect-video rounded-lg flex items-center justify-center border border-gray-200 dark:border-gray-700"
              >
                <div className="text-center">
                  <div className="text-gray-400 dark:text-gray-500 text-sm mb-2">{title}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-600">Placeholder for image</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default Cozmoclench;
