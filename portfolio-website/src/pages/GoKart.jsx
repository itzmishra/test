import { motion } from 'framer-motion';
import { ArrowLeft, Zap, Settings, Wrench, Award } from 'lucide-react';
import { Link } from 'react-router-dom';
import SectionTitle from '../components/SectionTitle';

const GoKart = () => {
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
            title="Formula Student Go-Kart Project"
            subtitle="Formula Student | Vehicle Design"
          />
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mt-6"
          >
            Complete design and development of a Formula Student go-kart with focus on acceleration events, 
            chassis optimization, and engine tuning for maximum performance.
          </motion.p>
        </div>
      </section>

      {/* Overview Section */}
      <section className="section-padding bg-white dark:bg-gray-800">
        <div className="container-custom">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">Project Overview</h2>
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed max-w-3xl">
            As part of the Formula Student team, I led the design and development of a high-performance go-kart 
            optimized for acceleration events. The project involved comprehensive engineering work including chassis 
            design, engine selection and tuning, drivetrain optimization, and overall vehicle dynamics.
          </p>
        </div>
      </section>

      {/* Key Sections */}
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
                <Zap className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                  Acceleration Event Design
                </h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed">
                Designed and optimized the vehicle specifically for acceleration events, focusing on power-to-weight ratio, 
                traction optimization, and launch characteristics. Implemented advanced suspension tuning and weight 
                distribution strategies.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center gap-3 mb-4">
                <Award className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                  Points Distribution Strategy
                </h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed">
                Developed a comprehensive strategy for maximizing competition points across all events including 
                acceleration, autocross, endurance, and design events. Balanced vehicle design to excel in multiple 
                disciplines.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center gap-3 mb-4">
                <Settings className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                  Chassis Design
                </h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed">
                Designed a lightweight yet rigid chassis using CAD software (SolidWorks) and performed FEA analysis 
                using ANSYS. Optimized for strength, weight reduction, and manufacturability while meeting all safety 
                regulations.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center gap-3 mb-4">
                <Wrench className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                  Engine Design & Tuning
                </h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed">
                Selected and tuned the engine for optimal performance in acceleration events. Worked on intake/exhaust 
                optimization, ECU tuning, and drivetrain matching to maximize power delivery and efficiency.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Responsibilities Section */}
      <section className="section-padding bg-white dark:bg-gray-800">
        <div className="container-custom">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">Key Responsibilities</h2>
          <div className="space-y-4">
            {[
              'Led the chassis design team and coordinated with other subsystems',
              'Performed FEA analysis and structural optimization',
              'Managed project timeline and ensured competition readiness',
              'Collaborated with engine and drivetrain teams for system integration',
              'Conducted testing and validation of design iterations',
              'Presented design concepts and technical reports to judges',
            ].map((responsibility, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="flex items-start gap-3"
              >
                <div className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full mt-2 flex-shrink-0" />
                <p className="text-gray-600 dark:text-gray-300">{responsibility}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Outcomes Section */}
      <section className="section-padding bg-gray-50 dark:bg-gray-900">
        <div className="container-custom">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">Project Outcomes</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { title: 'Weight Reduction', value: '15%', desc: 'Reduced chassis weight through optimization' },
              { title: 'Performance', value: 'Top 10', desc: 'Ranked in top 10 for acceleration events' },
              { title: 'Design Award', value: 'Finalist', desc: 'Finalist in design event category' },
            ].map((outcome, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 text-center"
              >
                <div className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                  {outcome.value}
                </div>
                <div className="font-semibold text-gray-900 dark:text-white mb-1">
                  {outcome.title}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-300">
                  {outcome.desc}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Placeholder for Images */}
      <section className="section-padding bg-white dark:bg-gray-800">
        <div className="container-custom">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">Project Gallery</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {['Chassis Design', 'Engine Setup', 'Final Assembly'].map((title, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                className="bg-gray-100 dark:bg-gray-700 aspect-video rounded-lg flex items-center justify-center"
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

export default GoKart;
