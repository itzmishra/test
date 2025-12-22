import { motion } from 'framer-motion';
import { ArrowLeft, Bot, Plane, Target, Zap } from 'lucide-react';
import { Link } from 'react-router-dom';
import SectionTitle from '../components/SectionTitle';

const Robotics = () => {
  const projects = [
    {
      title: 'Robosoccer Robot',
      icon: Bot,
      description: 'Designed and built an autonomous soccer-playing robot for robotics competitions. The robot features advanced navigation, ball detection, and strategic gameplay algorithms.',
      features: [
        'Autonomous navigation system',
        'Computer vision for ball detection',
        'Multi-sensor fusion',
        'Strategic decision-making algorithms',
      ],
    },
    {
      title: 'Aeromodelling',
      icon: Plane,
      description: 'Developed and built radio-controlled aircraft models with focus on aerodynamics, stability, and control systems. Participated in aeromodelling competitions.',
      features: [
        'Aerodynamic design and analysis',
        'Flight control systems',
        'Radio control integration',
        'Competition participation',
      ],
    },
    {
      title: 'Autonomous Navigation Systems',
      icon: Target,
      description: 'Worked on various autonomous navigation projects including path planning, obstacle avoidance, and sensor integration for mobile robots.',
      features: [
        'Path planning algorithms',
        'Obstacle detection and avoidance',
        'Sensor fusion (LiDAR, IMU, cameras)',
        'Real-time control systems',
      ],
    },
  ];

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
            title="Robotics Portfolio"
            subtitle="Robotics | Multiple Projects"
          />
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mt-6"
          >
            A collection of robotics projects showcasing expertise in autonomous systems, mechanism design, 
            and control systems. From competitive robots to research applications.
          </motion.p>
        </div>
      </section>

      {/* Projects Grid */}
      <section className="section-padding bg-white dark:bg-gray-800">
        <div className="container-custom">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {projects.map((project, index) => {
              const Icon = project.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl border border-gray-200 dark:border-gray-700 card-hover"
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 bg-blue-100 dark:bg-blue-900 rounded-lg">
                      <Icon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                      {project.title}
                    </h3>
                  </div>
                  <p className="text-gray-600 dark:text-gray-300 text-sm mb-4 leading-relaxed">
                    {project.description}
                  </p>
                  <ul className="space-y-2">
                    {project.features.map((feature, featureIndex) => (
                      <li
                        key={featureIndex}
                        className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-300"
                      >
                        <Zap className="w-4 h-4 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Additional Projects Section */}
      <section className="section-padding bg-gray-50 dark:bg-gray-900">
        <div className="container-custom">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">Additional Robotics Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[
              {
                title: 'Line Following Robot',
                desc: 'Developed a line-following robot with PID control for precise navigation along predefined paths.',
              },
              {
                title: 'Obstacle Avoidance Robot',
                desc: 'Built an autonomous robot capable of navigating through environments while avoiding obstacles using ultrasonic sensors.',
              },
              {
                title: 'Pick and Place Robot',
                desc: 'Designed a robotic arm system for pick and place operations with precision control and object recognition.',
              },
              {
                title: 'Swarm Robotics',
                desc: 'Research project on coordinated behavior in multi-robot systems for collaborative task execution.',
              },
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700"
              >
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">{item.title}</h3>
                <p className="text-sm text-gray-600 dark:text-gray-300">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Skills & Technologies */}
      <section className="section-padding bg-white dark:bg-gray-800">
        <div className="container-custom">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">Technologies & Skills</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              'ROS (Robot Operating System)',
              'Arduino',
              'Raspberry Pi',
              'Python',
              'C++',
              'Computer Vision',
              'Sensor Fusion',
              'Control Systems',
              'CAD Design',
              '3D Printing',
              'Electronics',
              'Embedded Systems',
            ].map((tech, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg text-center border border-gray-200 dark:border-gray-700"
              >
                <span className="text-sm font-medium text-gray-900 dark:text-white">{tech}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Placeholder for Images */}
      <section className="section-padding bg-gray-50 dark:bg-gray-900">
        <div className="container-custom">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">Project Gallery</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {['Robosoccer Robot', 'Aeromodelling Aircraft', 'Navigation System'].map((title, index) => (
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

export default Robotics;
