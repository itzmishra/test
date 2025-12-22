import { motion } from 'framer-motion';
import { Briefcase, GraduationCap, Code2, Cpu, Wrench, Tool } from 'lucide-react';
import profileData from '../data/profile.json';
import SectionTitle from '../components/SectionTitle';

const About = () => {
  const skillIcons = {
    engineering: Wrench,
    machineLearning: Cpu,
    robotics: Code2,
    programming: Code2,
    tools: Tool,
  };

  return (
    <div className="min-h-screen pt-20">
      {/* Hero Section */}
      <section className="section-padding bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800">
        <div className="container-custom">
          <SectionTitle title="About Me" subtitle="Get to Know Me" center />
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-3xl mx-auto mt-8"
          >
            <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed text-center">
              {profileData.bio}
            </p>
          </motion.div>
        </div>
      </section>

      {/* Experience Section */}
      <section className="section-padding bg-white dark:bg-gray-800">
        <div className="container-custom">
          <SectionTitle title="Experience" subtitle="Professional Journey" />
          <div className="mt-12 space-y-8">
            {profileData.experience.map((exp, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="relative pl-8 border-l-4 border-blue-600 dark:border-blue-400"
              >
                <div className="absolute -left-3 top-0 w-6 h-6 bg-blue-600 dark:bg-blue-400 rounded-full border-4 border-white dark:border-gray-800" />
                <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-lg">
                  <div className="flex items-center gap-3 mb-2">
                    <Briefcase className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                      {exp.role}
                    </h3>
                  </div>
                  <p className="text-blue-600 dark:text-blue-400 font-semibold mb-2">
                    {exp.organization} • {exp.period}
                  </p>
                  <p className="text-gray-600 dark:text-gray-300">
                    {exp.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Skills Section */}
      <section className="section-padding bg-gray-50 dark:bg-gray-900">
        <div className="container-custom">
          <SectionTitle title="Skills & Expertise" subtitle="What I Do" />
          <div className="mt-12 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.entries(profileData.skills).map(([category, skills], index) => {
              const Icon = skillIcons[category] || Code2;
              return (
                <motion.div
                  key={category}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700"
                >
                  <div className="flex items-center gap-3 mb-4">
                    <Icon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white capitalize">
                      {category.replace(/([A-Z])/g, ' $1').trim()}
                    </h3>
                  </div>
                  <ul className="space-y-2">
                    {skills.map((skill, skillIndex) => (
                      <li
                        key={skillIndex}
                        className="text-gray-600 dark:text-gray-300 text-sm flex items-center gap-2"
                      >
                        <span className="w-1.5 h-1.5 bg-blue-600 dark:bg-blue-400 rounded-full" />
                        {skill}
                      </li>
                    ))}
                  </ul>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Education Section */}
      <section className="section-padding bg-white dark:bg-gray-800">
        <div className="container-custom">
          <SectionTitle title="Education" subtitle="Academic Background" />
          <div className="mt-12 space-y-6">
            {profileData.education.map((edu, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="bg-gray-50 dark:bg-gray-900 p-6 rounded-lg border border-gray-200 dark:border-gray-700"
              >
                <div className="flex items-center gap-3 mb-2">
                  <GraduationCap className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                    {edu.degree} in {edu.field}
                  </h3>
                </div>
                <p className="text-blue-600 dark:text-blue-400 font-semibold mb-2">
                  {edu.institution} • {edu.period}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default About;
